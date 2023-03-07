# Copyright (c) 2022-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from dask.distributed import wait, default_client
import cugraph.dask.comms.comms as Comms
import dask_cudf
import cudf
from cugraph.dask.common.input_utils import get_distributed_data
from cugraph.utilities import renumber_vertex_pair

from pylibcugraph.experimental import (
    jaccard_coefficients as pylibcugraph_jaccard_coefficients,
)
from pylibcugraph import ResourceHandle


def convert_to_cudf(cp_arrays):
    """
    Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
    """

    cupy_first, cupy_second, cupy_similarity = cp_arrays

    df = cudf.DataFrame()
    df["first"] = cupy_first
    df["second"] = cupy_second
    df["jaccard_coeff"] = cupy_similarity

    return df


def _call_plc_jaccard(
    sID, mg_graph_x, vertex_pair, use_weight, do_expensive_check, vertex_pair_col_name
):

    first = vertex_pair[vertex_pair_col_name[0]]
    second = vertex_pair[vertex_pair_col_name[1]]

    return pylibcugraph_jaccard_coefficients(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        first=first,
        second=second,
        use_weight=use_weight,
        do_expensive_check=do_expensive_check,
    )


def jaccard(input_graph, vertex_pair=None, use_weight=False):
    """
    Compute the Jaccard similarity between each pair of vertices connected by
    an edge, or between arbitrary pairs of vertices specified by the user.
    Jaccard similarity is defined between two sets as the ratio of the volume
    of their intersection divided by the volume of their union. In the context
    of graphs, the neighborhood of a vertex is seen as a set. The Jaccard
    similarity weight of each edge represents the strength of connection
    between vertices based on the relative similarity of their neighbors. If
    first is specified but second is not, or vice versa, an exception will be
    thrown.

    NOTE: If the vertex_pair parameter is not specified then the behavior
    of cugraph.jaccard is different from the behavior of
    networkx.jaccard_coefficient.

    cugraph.dask.jaccard, in the absence of a specified vertex pair list, will
    compute the two_hop_neighbors of the entire graph to construct a vertex pair
    list and will return the jaccard coefficient for those vertex pairs. This is
    not advisable as the vertex_pairs can grow exponentially with respect to the
    size of the datasets

    networkx.jaccard_coefficient, in the absence of a specified vertex
    pair list, will return an upper triangular dense matrix, excluding
    the diagonal as well as vertex pairs that are directly connected
    by an edge in the graph, of jaccard coefficients.  Technically, networkx
    returns a lazy iterator across this upper triangular matrix where
    the actual jaccard coefficient is computed when the iterator is
    dereferenced.  Computing a dense matrix of results is not feasible
    if the number of vertices in the graph is large (100,000 vertices
    would result in 4.9 billion values in that iterator).

    If your graph is small enough (or you have enough memory and patience)
    you can get the interesting (non-zero) values that are part of the networkx
    solution by doing the following:

    But please remember that cugraph will fill the dataframe with the entire
    solution you request, so you'll need enough memory to store the 2-hop
    neighborhood dataframe.


    Parameters
    ----------
    input_graph : cugraph.Graph
        cuGraph Graph instance, should contain the connectivity information
        as an edge list (edge weights are not supported yet for this algorithm). The
        graph should be undirected where an undirected edge is represented by a
        directed edge in both direction. The adjacency list will be computed if
        not already present.

        This implementation only supports undirected, unweighted Graph.

    vertex_pair : cudf.DataFrame, optional (default=None)
        A GPU dataframe consisting of two columns representing pairs of
        vertices. If provided, the jaccard coefficient is computed for the
        given vertex pairs.  If the vertex_pair is not provided then the
        current implementation computes the jaccard coefficient for all
        adjacent vertices in the graph.

    use_weight : bool, optional (default=False)
        Currently not supported

    Returns
    -------
    result : dask_cudf.DataFrame
        GPU distributed data frame containing 2 dask_cudf.Series

        ddf['first']: dask_cudf.Series
            The first vertex ID of each pair (will be identical to first if specified).
        ddf['second']: dask_cudf.Series
            The second vertex ID of each pair (will be identical to second if
            specified).
        ddf['jaccard_coeff']: dask_cudf.Series
            The computed jaccard coefficient between the first and the second
            vertex ID.
    """

    if input_graph.is_directed():
        raise ValueError("input graph must be undirected")

    if vertex_pair is None:
        # Call two_hop neighbor of the entire graph
        vertex_pair = input_graph.get_two_hop_neighbors()

    vertex_pair_col_name = vertex_pair.columns

    if use_weight:
        raise ValueError("'use_weight' is currently not supported.")

    if input_graph.is_weighted():
        raise ValueError("Weighted graphs are currently not supported.")

    if isinstance(vertex_pair, (dask_cudf.DataFrame, cudf.DataFrame)):
        vertex_pair = renumber_vertex_pair(input_graph, vertex_pair)

    elif vertex_pair is not None:
        raise ValueError("vertex_pair must be a dask_cudf or cudf dataframe")

    if not isinstance(vertex_pair, (dask_cudf.DataFrame)):
        vertex_pair = dask_cudf.from_cudf(
            vertex_pair, npartitions=len(Comms.get_workers())
        )
    vertex_pair = get_distributed_data(vertex_pair)
    wait(vertex_pair)
    vertex_pair = vertex_pair.worker_to_parts

    # Initialize dask client
    client = default_client()

    do_expensive_check = False

    if vertex_pair is not None:
        result = [
            client.submit(
                _call_plc_jaccard,
                Comms.get_session_id(),
                input_graph._plc_graph[w],
                vertex_pair[w][0],
                use_weight,
                do_expensive_check,
                vertex_pair_col_name,
                workers=[w],
                allow_other_workers=False,
            )
            for w in Comms.get_workers()
        ]

    wait(result)

    cudf_result = [client.submit(convert_to_cudf, cp_arrays) for cp_arrays in result]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result).persist()
    wait(ddf)

    # Wait until the inactive futures are released
    wait([(r.release(), c_r.release()) for r, c_r in zip(result, cudf_result)])

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "first")
        ddf = input_graph.unrenumber(ddf, "second")

    return ddf
