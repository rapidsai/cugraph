# Copyright (c) 2021, NVIDIA CORPORATION.
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
from cugraph.dask.common.input_utils import (get_distributed_data,
                                             get_vertex_partition_offsets)
from cugraph.dask.components import mg_connectivity_wrapper as mg_connectivity
import cugraph.comms.comms as Comms
import dask_cudf


def call_wcc(sID,
             data,
             num_verts,
             num_edges,
             vertex_partition_offsets):
    wid = Comms.get_worker_id(sID)
    handle = Comms.get_handle(sID)
    return mg_connectivity.mg_wcc(data[0],
                                  num_verts,
                                  num_edges,
                                  vertex_partition_offsets,
                                  wid,
                                  handle)


def weakly_connected_components(input_graph):

    client = default_client()

    input_graph.compute_renumber_edge_list()

    ddf = input_graph.edgelist.edgelist_df
    vertex_partition_offsets = get_vertex_partition_offsets(input_graph)
    num_verts = vertex_partition_offsets.iloc[-1]
    num_edges = len(ddf)
    data = get_distributed_data(ddf)

    result = [client.submit(call_wcc,
                            Comms.get_session_id(),
                            wf[1],
                            num_verts,
                            num_edges,
                            vertex_partition_offsets,
                            workers=[wf[0]])
              for idx, wf in enumerate(data.worker_to_parts.items())]
    wait(result)
    ddf = dask_cudf.from_delayed(result)

    if input_graph.renumbered:
        return input_graph.unrenumber(ddf, 'vertex')

    return ddf
