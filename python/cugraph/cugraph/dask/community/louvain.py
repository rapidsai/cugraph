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
import operator as op

from pylibcugraph import ResourceHandle
from pylibcugraph import louvain as pylibcugraph_louvain


def convert_to_cudf(cupy_vertex, cupy_partition):
    """
    Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
    """
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertex
    df["partition"] = cupy_partition

    return df


def _call_plc_louvain(sID, mg_graph_x, max_iter, resolution, do_expensive_check):
    return pylibcugraph_louvain(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        max_level=max_iter,
        resolution=resolution,
        do_expensive_check=do_expensive_check,
    )


def louvain(input_graph, max_iter=100, resolution=1.0):
    """
    Compute the modularity optimizing partition of the input graph using the
    Louvain method

    It uses the Louvain method described in:

    VD Blondel, J-L Guillaume, R Lambiotte and E Lefebvre: Fast unfolding of
    community hierarchies in large networks, J Stat Mech P10008 (2008),
    http://arxiv.org/abs/0803.0476

    Parameters
    ----------
    G : cugraph.Graph
        The graph descriptor should contain the connectivity information
        and weights. The adjacency list will be computed if not already
        present.
        The current implementation only supports undirected graphs.

    max_iter : integer, optional (default=100)
        This controls the maximum number of levels/iterations of the Louvain
        algorithm. When specified the algorithm will terminate after no more
        than the specified number of iterations. No error occurs when the
        algorithm terminates early in this manner.

    resolution: float/double, optional (default=1.0)
        Called gamma in the modularity formula, this changes the size
        of the communities.  Higher resolutions lead to more smaller
        communities, lower resolutions lead to fewer larger communities.
        Defaults to 1.

    Returns
    -------
    parts : dask_cudf.DataFrame
        GPU data frame of size V containing two columns the vertex id and the
        partition id it is assigned to.

        ddf['vertex'] : cudf.Series
            Contains the vertex identifiers
        ddf['partition'] : cudf.Series
            Contains the partition assigned to the vertices

    modularity_score : float
        a floating point number containing the global modularity score of the
        partitioning.

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> parts = cugraph.louvain(G)

    """

    if input_graph.is_directed():
        raise ValueError("input graph must be undirected")

    # Initialize dask client
    client = default_client()

    do_expensive_check = False

    result = [
        client.submit(
            _call_plc_louvain,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            max_iter,
            resolution,
            do_expensive_check,
            workers=[w],
            allow_other_workers=False,
        )
        for w in Comms.get_workers()
    ]

    wait(result)

    # futures is a list of Futures containing tuples of (DataFrame, mod_score),
    # unpack using separate calls to client.submit with a callable to get
    # individual items.
    # FIXME: look into an alternate way (not returning a tuples, accessing
    # tuples differently, etc.) since multiple client.submit() calls may not be
    # optimal.
    result_vertex = [client.submit(op.getitem, f, 0) for f in result]
    result_partition = [client.submit(op.getitem, f, 1) for f in result]
    mod_score = [client.submit(op.getitem, f, 2) for f in result]

    cudf_result = [
        client.submit(convert_to_cudf, cp_vertex_arrays, cp_partition_arrays)
        for cp_vertex_arrays, cp_partition_arrays in zip(
            result_vertex, result_partition
        )
    ]

    wait(cudf_result)
    # Each worker should have computed the same mod_score
    mod_score = mod_score[0].result()

    ddf = dask_cudf.from_delayed(cudf_result).persist()
    wait(ddf)

    # Wait until the inactive futures are released
    wait([(r.release(), c_r.release()) for r, c_r in zip(result, cudf_result)])

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "vertex")

    return ddf, mod_score
