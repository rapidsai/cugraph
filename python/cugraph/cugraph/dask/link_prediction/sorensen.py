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
    sorensen_coefficients as pylibcugraph_sorensen_coefficients,
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
    df["sorensen_coeff"] = cupy_similarity

    return df


def _call_plc_sorensen(
    sID, mg_graph_x, vertex_pair, use_weight, do_expensive_check, vertex_pair_col_name
):

    first = vertex_pair[vertex_pair_col_name[0]]
    second = vertex_pair[vertex_pair_col_name[1]]

    return pylibcugraph_sorensen_coefficients(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        first=first,
        second=second,
        use_weight=use_weight,
        do_expensive_check=do_expensive_check,
    )


def sorensen(input_graph, vertex_pair=None, use_weight=False):
    """
    Compute the Sorensen coefficient between each pair of vertices connected by
    an edge, or between arbitrary pairs of vertices specified by the user.
    Sorensen coefficient is defined between two sets as the ratio of twice the
    volume of their intersection divided by the volume of each set.
    If first is specified but second is not, or vice versa, an exception will
    be thrown.

    cugraph.dask.sorensen, in the absence of a specified vertex pair list, will
    compute the two_hop_neighbors of the entire graph to construct a vertex pair
    list and will return the sorensen coefficient for those vertex pairs. This is
    not advisable as the vertex_pairs can grow exponentially with respect to the
    size of the datasets

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
        vertices. If provided, the sorensen coefficient is computed for the
        given vertex pairs.  If the vertex_pair is not provided then the
        current implementation computes the sorensen coefficient for all
        adjacent vertices in the graph.

    use_weight : bool, optional (default=False)
        Currently not supported

    Returns
    -------
    result : dask_cudf.DataFrame
        GPU distributed data frame containing 2 dask_cudf.Series

        ddf['first']: dask_cudf.Series
            The first vertex ID of each pair(will be identical to first if specified).
        ddf['second']: dask_cudf.Series
            The second vertex ID of each pair(will be identical to second if
            specified).
        ddf['sorensen_coeff']: dask_cudf.Series
            The computed sorensen coefficient between the first and the second
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
                _call_plc_sorensen,
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
