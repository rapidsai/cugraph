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
from cugraph.dask.common.input_utils import (get_distributed_data,
                                             get_vertex_partition_offsets)
from pylibcugraph.experimental import (ResourceHandle,
                                       GraphProperties,
                                       MGGraph,
                                       katz_centrality as pylibcugraph_katz
                                       )
import cugraph.dask.comms.comms as Comms
import dask_cudf
import cudf
import cupy


def call_katz_centrality(sID,
                         data,
                         graph_properties,
                         store_transposed,
                         do_expensive_check,
                         src_col_name,
                         dst_col_name,
                         num_verts,
                         num_edges,
                         vertex_partition_offsets,
                         aggregate_segment_offsets,
                         alpha,
                         beta,
                         max_iter,
                         tol,
                         nstart,
                         normalized):
    handle = Comms.get_handle(sID)
    h = ResourceHandle(handle.getHandle())
    srcs = data[0][src_col_name]
    dsts = data[0][dst_col_name]
    weights = cudf.Series(cupy.ones(srcs.size, dtype="float32"))

    if "value" in data[0].columns:
        weights = data[0]['value']

    initial_hubs_guess_values = None
    if nstart:
        initial_hubs_guess_values = nstart["values"]

    mg = MGGraph(h,
                 graph_properties,
                 srcs,
                 dsts,
                 weights,
                 store_transposed,
                 num_edges,
                 do_expensive_check)

    result = pylibcugraph_katz(h,
                               mg,
                               initial_hubs_guess_values,
                               alpha,
                               beta,
                               tol,
                               max_iter,
                               do_expensive_check)
    return result


def convert_to_cudf(cp_arrays):
    """
    create a cudf DataFrame from cupy arrays
    """
    cupy_vertices, cupy_values = cp_arrays
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertices
    df["katz_centrality"] = cupy_values
    return df


def katz_centrality(
    input_graph, alpha=None, beta=1.0, max_iter=100, tol=1.0e-6,
    nstart=None, normalized=True
):
    client = default_client()

    graph_properties = GraphProperties(
        is_multigraph=False)

    store_transposed = False
    do_expensive_check = False

    src_col_name = input_graph.renumber_map.renumbered_src_col_name
    dst_col_name = input_graph.renumber_map.renumbered_dst_col_name

    # FIXME Move this call to the function creating a directed
    # graph from a dask dataframe because duplicated edges need
    # to be dropped
    ddf = input_graph.edgelist.edgelist_df
    ddf = ddf.map_partitions(
        lambda df: df.drop_duplicates(subset=[src_col_name, dst_col_name]))

    num_edges = len(ddf)
    data = get_distributed_data(ddf)

    input_graph.compute_renumber_edge_list(transposed=True)
    vertex_partition_offsets = get_vertex_partition_offsets(input_graph)
    num_verts = vertex_partition_offsets.iloc[-1]

    cupy_result = [client.submit(call_katz_centrality,
                                 Comms.get_session_id(),
                                 wf[1],
                                 graph_properties,
                                 store_transposed,
                                 do_expensive_check,
                                 src_col_name,
                                 dst_col_name,
                                 num_verts,
                                 num_edges,
                                 vertex_partition_offsets,
                                 input_graph.aggregate_segment_offsets,
                                 alpha,
                                 beta,
                                 max_iter,
                                 tol,
                                 nstart,
                                 normalized,
                                 workers=[wf[0]])
                   for idx, wf in enumerate(data.worker_to_parts.items())]

    wait(cupy_result)

    cudf_result = [client.submit(convert_to_cudf,
                                 cp_arrays,
                                 workers=client.who_has(
                                     cp_arrays)[cp_arrays.key])
                   for cp_arrays in cupy_result]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result)
    if input_graph.renumbered:
        return input_graph.unrenumber(ddf, 'vertex')

    return ddf
