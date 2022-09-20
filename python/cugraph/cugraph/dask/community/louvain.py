# Copyright (c) 2022, NVIDIA CORPORATION.
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

from dask.distributed import wait
import cugraph.dask.comms.comms as Comms
import dask_cudf
import cudf

from pylibcugraph import ResourceHandle
from pylibcugraph import louvain as pylibcugraph_louvain


def convert_to_cudf(cp_arrays):
    """
    Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
    """
    cupy_vertices, cupy_partition = cp_arrays
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertices
    df["partition"] = cupy_partition

    return df


def _call_plc_louvain(sID,
                      mg_graph_x,
                      max_iter,
                      resolution,
                      do_expensive_check):
    return pylibcugraph_louvain(
        resource_handle=ResourceHandle(
            Comms.get_handle(sID).getHandle()
        ),
        graph=mg_graph_x,
        max_level=max_iter,
        resolution=resolution,
        do_expensive_check=do_expensive_check
    )


def louvain(input_graph, max_iter=100, resolution=1.):
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
    parts : cudf.DataFrame
        GPU data frame of size V containing two columns the vertex id and the
        partition id it is assigned to.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['partition'] : cudf.Series
            Contains the partition assigned to the vertices

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> parts = cugraph.louvain(G)

    """

    if input_graph.is_directed():
        raise ValueError("input graph must be undirected")

    # Initialize dask client
    client = input_graph._client

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

    cudf_result = [client.submit(convert_to_cudf,
                                 cp_arrays)
                   for cp_arrays in result]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result).persist()
    wait(ddf)

    # Wait until the inactive futures are released
    wait([(r.release(), c_r.release())
         for r, c_r in zip(result, cudf_result)])

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "vertex")

    return ddf
