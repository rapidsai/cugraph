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

from pylibcugraph import ResourceHandle, triangle_count as pylibcugraph_triangle_count


def _call_triangle_count(
    sID,
    mg_graph_x,
    start_list,
    do_expensive_check,
):

    return pylibcugraph_triangle_count(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        start_list=start_list,
        do_expensive_check=do_expensive_check,
    )


def convert_to_cudf(cp_arrays):
    """
    Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
    """
    cupy_vertices, cupy_counts = cp_arrays
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertices
    df["counts"] = cupy_counts

    return df


def triangle_count(input_graph, start_list=None):
    """
    Computes the number of triangles (cycles of length three) and the number
    per vertex in the input graph.

    Parameters
    ----------
    input_graph : cugraph.graph
        cuGraph graph descriptor, should contain the connectivity information,
        (edge weights are not used in this algorithm).
        The current implementation only supports undirected graphs.

    start_list : list or cudf.Series
        list of vertices for triangle count. if None the entire set of vertices
        in the graph is processed


    Returns
    -------
    result : dask_cudf.DataFrame
        GPU distributed data frame containing 2 dask_cudf.Series

    ddf['vertex']: dask_cudf.Series
            Contains the triangle counting vertices
    ddf['counts']: dask_cudf.Series
        Contains the triangle counting counts
    """
    if input_graph.is_directed():
        raise ValueError("input graph must be undirected")
    # Initialize dask client
    client = default_client()

    if start_list is not None:
        if isinstance(start_list, int):
            start_list = [start_list]
        if isinstance(start_list, list):
            start_list = cudf.Series(start_list)
        if not isinstance(start_list, cudf.Series):
            raise TypeError(
                f"'start_list' must be either a list or a cudf.Series,"
                f"got: {start_list.dtype}"
            )

        # start_list uses "external" vertex IDs, but since the graph has been
        # renumbered, the start vertex IDs must also be renumbered.
        if input_graph.renumbered:
            start_list = input_graph.lookup_internal_vertex_id(start_list).compute()

    do_expensive_check = False

    result = [
        client.submit(
            _call_triangle_count,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            start_list,
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
