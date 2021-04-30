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

from dask.distributed import wait, default_client, Client

#from cugraph.dask.common.input_utils #create some utils functions

#from cugraph.dask.utilities import\
#    mg_generate_edgelist_wrapper as mg_generate_edgelist
import cugraph.comms.comms as Comms
from cugraph.utilities.graph_generator import graph_generator_edgelist as graph_generator
import dask_cudf


def calc_num_edges_per_worker(num_workers, num_edges):
    #48 and 10
    L= []
    w = num_edges//num_workers
    r = num_edges%num_workers
    for i in range (num_workers):
        if (i<r):
            L.append(w+1)
        else:
            L.append(w)
    return L



def graph_generator_edgelist(scale, 
                             num_edges,
                             a,
                             b,
                             c,
                             seed,
                             clip_and_flip,
                             scramble_vertex_ids):
    

    
    #client = default_client()
    client = Client() #change this
    num_workers = len(client.scheduler_info()['workers'])

    list_job = calc_num_edges_per_worker(num_workers, num_edges)

    #edges = get_distributed_data() #call function to distribute the edge generation 
    #78 10
 
    L=[client.submit(graph_generator,
                               scale, 
                               n_edges,
                               a,
                               b,
                               c,
                               seed,
                               clip_and_flip,
                               scramble_vertex_ids) for seed, n_edges in enumerate(list_job)]
    
    
    #client.gather(L)

    return L
