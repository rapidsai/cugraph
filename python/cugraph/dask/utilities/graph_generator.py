# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
from cugraph.dask.common.input_utils #create some utils functions
from cugraph.dask.centrality import\
    mg_generate_edgelist_wrapper as mg_generate_edgelist_wrapper
import cugraph.comms.comms as Comms
import dask_cudf


def call_generate_edgelist(sID,
                            data,
                            scale, 
                            num_edges,
                            a,
                            b,
                            c,
                            seed,
                            clip_and_flip,
                            scramble_vertex_ids):
    wid = Comms.get_worker_id(sID)
    handle = Comms.get_handle(sID)
    return mg_graph_generator_edgelist.mg_graph_generator_edgelist(data[0],
                                                 num_verts,
                                                 num_edges,
                                                 vertex_partition_offsets,
                                                 wid,
                                                 handle,
                                                 alpha,
                                                 beta,
                                                 max_iter,
                                                 tol,
                                                 nstart,
                                                 normalized)


def graph_generator_edgelist(input_graph,
                    alpha=None,
                    beta=None,
                    max_iter=100,
                    tol=1.0e-5,
                    nstart=None,
                    normalized=True):
    


    client = default_client()


    edges = get_distributed_data() #call function to distribute the edge generation 
    
    ddf = [client.submit(graph_generator_edgelist,
                            Comms.get_session_id(),
                            wf[1],
                            scale, 
                            num_edges,
                            a,
                            b,
                            c,
                            seed,
                            clip_and_flip,
                            scramble_vertex_ids,
                            workers=[wf[0]])
                for wf in (edges)]   #determine this parameter
    wait(ddf)
    ddf = dask_cudf.from_delayed(ddf)

    return ddf
