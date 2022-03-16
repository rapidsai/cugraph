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

# from dask.distributed import wait, default_client
# from cugraph.dask.common.input_utils import (get_distributed_data,
#                                              get_vertex_partition_offsets)
# import cugraph.comms.comms as Comms
# import dask_cudf
import pylibcugraph.experimental as pylibcugraph


def uniform_neighborhood(G,
                         start_info_list,
                         fanout_vals,
                         with_replacement=True):
    """
    Does neighborhood sampling.
    Parameters
    ----------
    G : cugraph.DiGraph
        cuGraph graph, which contains connectivity information as dask cudf
        edge list dataframe

    start_info_list : list
        List of starting vertices for neighborhood sampling

    fanout_vals : list
        List of branching out (fan-out) degrees per starting vertex for each
        hop level

    with_replacement: bool, optional (default=True)
        Flag to specify if the random sampling is done with replacement
    """
    print("Hello from cugraph/dask!")
    """
    # Initialize dask client
    client = default_client()
    # Important for handling renumbering
    input_graph.compute_renumber_edge_list(transposed=True)

    ddf = input_graph.edgelist.edgelist_df
    vertex_partition_offsets = get_vertex_partition_offsets(input_graph)
    num_verts = vertex_partition_offsets.iloc[-1]
    num_edges = len(ddf)
    data = get_distributed_data(ddf)
    """

    # Would want G or whatever takes its place to be a pylibcugraph MG Graph,
    # which isn't implemented yet but will be made. The pylib MG Graph will use
    # cugraph_mg_graph_create, which means that #2110 is a dependency
    return pylibcugraph.uniform_neighborhood_sampling(G, start_info_list,
                                                      fanout_vals,
                                                      with_replacement)
