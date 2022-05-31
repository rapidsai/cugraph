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

from dask.distributed import wait, default_client
from cugraph.dask.common.input_utils import get_distributed_data

import cugraph.dask.comms.comms as Comms
import dask_cudf
import cudf

from pylibcugraph.experimental import triangle_count as \
    pylibcugraph_triangle_count

from pylibcugraph import (ResourceHandle,
                          GraphProperties,
                          MGGraph
                          )


def call_triangles(sID,
                   data,
                   src_col_name,
                   dst_col_name,
                   graph_properties,
                   store_transposed,
                   num_edges,
                   do_expensive_check,
                   start_list
                   ):

    handle = Comms.get_handle(sID)
    h = ResourceHandle(handle.getHandle())
    srcs = data[0][src_col_name]
    dsts = data[0][dst_col_name]
    weights = None
    if "value" in data[0].columns:
        weights = data[0]['value']

    mg = MGGraph(h,
                 graph_properties,
                 srcs,
                 dsts,
                 weights,
                 store_transposed,
                 num_edges,
                 do_expensive_check)

    result = pylibcugraph_triangle_count(h,
                                         mg,
                                         start_list,
                                         do_expensive_check)

    return result


def convert_to_cudf(cp_arrays):
    """
    Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
    """
    cupy_vertices, cupy_counts = cp_arrays
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertices
    df["counts"] = cupy_counts

    return df


def triangle_count(input_graph,
                   start_list=None):
    """
    Computes the number of triangles (cycles of length three) and the number
    per vertex in the input graph.

    Parameters
    ----------
    input_graph : cugraph.graph
        cuGraph graph descriptor, should contain the connectivity information,
        (edge weights are not used in this algorithm).
        The current implementation only supports undirected graphs.

    start_list : not supported
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
    # FIXME: start_list is disabled
    start_list = None
    if input_graph.is_directed():
        raise ValueError("input graph must be undirected")
    # Initialize dask client
    client = default_client()
    # In the future, once all the algos follow the C/Pylibcugraph path,
    # compute_renumber_edge_list will only be used for multicolumn and
    # string vertices since the renumbering will be done in pylibcugraph
    input_graph.compute_renumber_edge_list(
        transposed=False, legacy_renum_only=True)

    if start_list is not None:
        if isinstance(start_list, int):
            start_list = [start_list]
        if isinstance(start_list, list):
            start_list = cudf.Series(start_list)
            if start_list.dtype != 'int32':
                raise ValueError(f"'start_list' must have int32 values, "
                                 f"got: {start_list.dtype}")
        if not isinstance(start_list, cudf.Series):
            raise TypeError(
                    f"'start_list' must be either a list or a cudf.Series,"
                    f"got: {start_list.dtype}")

        # start_list uses "external" vertex IDs, but since the graph has been
        # renumbered, the start vertex IDs must also be renumbered.
        if input_graph.renumbered:
            start_list = input_graph.lookup_internal_vertex_id(
                start_list).compute()

    ddf = input_graph.edgelist.edgelist_df

    # FIXME: The parameter is_multigraph, store_transposed and
    # do_expensive_check must be derived from the input_graph.
    # For now, they are hardcoded.
    graph_properties = GraphProperties(
        is_symmetric=True, is_multigraph=False)
    store_transposed = False
    do_expensive_check = True

    num_edges = len(ddf)
    data = get_distributed_data(ddf)

    src_col_name = input_graph.renumber_map.renumbered_src_col_name
    dst_col_name = input_graph.renumber_map.renumbered_dst_col_name

    result = [client.submit(call_triangles,
                            Comms.get_session_id(),
                            wf[1],
                            src_col_name,
                            dst_col_name,
                            graph_properties,
                            store_transposed,
                            num_edges,
                            do_expensive_check,
                            start_list,
                            workers=[wf[0]])
              for idx, wf in enumerate(data.worker_to_parts.items())]

    wait(result)

    cudf_result = [client.submit(convert_to_cudf,
                                 cp_arrays)
                   for cp_arrays in result]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result)
    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "vertex")

    return ddf
