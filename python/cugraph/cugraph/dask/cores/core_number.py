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

from pylibcugraph import ResourceHandle, core_number as pylibcugraph_core_number


def convert_to_cudf(cp_arrays):
    """
    Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
    """
    cupy_vertices, cupy_core_number = cp_arrays
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertices
    df["core_number"] = cupy_core_number

    return df


def _call_plc_core_number(sID, mg_graph_x, dt_x, do_expensive_check):
    return pylibcugraph_core_number(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        degree_type=dt_x,
        do_expensive_check=do_expensive_check,
    )


def core_number(input_graph, degree_type="bidirectional"):
    """
    Compute the core numbers for the nodes of the graph G. A k-core of a graph
    is a maximal subgraph that contains nodes of degree k or more.
    A node has a core number of k if it belongs a k-core but not to k+1-core.
    This call does not support a graph with self-loops and parallel
    edges.

    Parameters
    ----------
    input_graph : cugraph.graph
        cuGraph graph descriptor, should contain the connectivity information,
        (edge weights are not used in this algorithm).
        The current implementation only supports undirected graphs.

    degree_type: str, (default="bidirectional")
        This option determines if the core number computation should be based
        on input, output, or both directed edges, with valid values being
        "incoming", "outgoing", and "bidirectional" respectively.


    Returns
    -------
    result : dask_cudf.DataFrame
        GPU distributed data frame containing 2 dask_cudf.Series

        ddf['vertex']: dask_cudf.Series
            Contains the core number vertices
        ddf['core_number']: dask_cudf.Series
            Contains the core number of vertices
    """

    if input_graph.is_directed():
        raise ValueError("input graph must be undirected")

    if degree_type not in ["incoming", "outgoing", "bidirectional"]:
        raise ValueError(
            f"'degree_type' must be either incoming, "
            f"outgoing or bidirectional, got: {degree_type}"
        )

    # Initialize dask client
    client = default_client()

    do_expensive_check = False

    result = [
        client.submit(
            _call_plc_core_number,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            degree_type,
            do_expensive_check,
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
        ddf = input_graph.unrenumber(ddf, "vertex")

    return ddf
