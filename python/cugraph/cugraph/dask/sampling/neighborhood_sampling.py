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

from dask.distributed import wait, default_client
from cugraph.dask.common.input_utils import get_distributed_data
import cugraph.comms.comms as Comms
import dask_cudf
import pylibcugraph.experimental as pylibcugraph


def call_nbr_sampling(sID,
                      data,
                      src_col_name,
                      dst_col_name,
                      num_edges,
                      do_expensive_check,
                      start_info_list,
                      h_fan_out,
                      with_replacement):

    # Preparation for graph creation
    handle = Comms.get_handle(sID)
    handle = pylibcugraph.experimental.ResourceHandle(handle.getHandle())
    graph_properties = pylibcugraph.experimental.GraphProperties(
        is_multigraph=False)
    srcs = data[0][src_col_name]
    dsts = data[0][dst_col_name]
    weights = None
    if "value" in data[0].columns:
        weights = data[0]['value']

    mg = pylibcugraph.MGGraph(handle,
                              graph_properties,
                              srcs,
                              dsts,
                              weights,
                              False,
                              num_edges,
                              do_expensive_check)

    return pylibcugraph.uniform_neighborhood_sampling(handle,
                                                      mg,
                                                      start_info_list,
                                                      h_fan_out,
                                                      with_replacement,
                                                      do_expensive_check)


def uniform_neighborhood(input_graph,
                         start_info_list,
                         fanout_vals,
                         with_replacement=True):
    """
    Does neighborhood sampling.

    Parameters
    ----------
    input_graph : cugraph.DiGraph
        cuGraph graph, which contains connectivity information as dask cudf
        edge list dataframe

    start_info_list : list
        ...

    fanout_vals : list
        List of branching out (fan-out) degrees per starting vertex for each
        hop level

    with_replacement: bool, optional (default=True)
        Flag to specify if the random sampling is done with replacement

    Returns
    -------
    result : dask_cudf.DataFrame
        GPU data frame containing two dask_cudf.Series

        ddf['srcs']: dask_cudf.Series
            Contains the source vertices from the sampling result
        ddf['dsts']: dask_cudf.Series
            Contains the destination vertices from the sampling result
        ddf['labels']: dask_cudf.Series
            Contains the start labels from the sampling result
        ddf['index']: dask_cudf.Series
            Contains the indices from the sampling result
        ddf['counts']: dask_cudf.Series
            Contains the transaction counts from the sampling result
    """

    print("Hello from cugraph/dask!")

    # Initialize dask client
    client = default_client()
    # Important for handling renumbering
    input_graph.compute_renumber_edge_list(transposed=True)

    ddf = input_graph.edgelist.edgelist_df
    # vertex_partition_offsets = get_vertex_partition_offsets(input_graph)
    # num_verts = vertex_partition_offsets.iloc[-1]
    num_edges = len(ddf)
    data = get_distributed_data(ddf)

    src_col_name = input_graph.renumber_map.renumbered_src_col_name
    dst_col_name = input_graph.renumber_map.renumbered_dst_col_name

    """
    # Would want G or whatever takes its place to be a pylibcugraph MG Graph,
    # which isn't implemented yet but will be made. The pylib MG Graph will use
    # cugraph_mg_graph_create, which means that #2110 is a dependency
    return pylibcugraph.uniform_neighborhood_sampling(G, start_info_list,
                                                      fanout_vals,
                                                      with_replacement)
    """
    result = [client.submit(call_nbr_sampling,
                            Comms.get_session_id(),
                            wf[1],
                            src_col_name,
                            dst_col_name,
                            num_edges,
                            False,
                            start_info_list,
                            fanout_vals,
                            with_replacement,
                            workers=[wf[0]])
              for idx, wf in enumerate(data.worker_to_parts.items())]

    wait(result)
    ddf = dask_cudf.from_delayed(result)
    if input_graph.renumbered:
        return input_graph.unrenumber(ddf, 'vertex')

    return ddf
