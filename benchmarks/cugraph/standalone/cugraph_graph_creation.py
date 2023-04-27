# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

from dask.distributed import wait
from cugraph.testing.mg_utils import generate_edgelist, get_allocation_counts_dask_persist, sizeof_fmt
from cugraph.testing.mg_utils import start_dask_client, stop_dask_client, enable_spilling
import cugraph
from cugraph.dask.comms import comms as Comms
from time import sleep

@get_allocation_counts_dask_persist
def construct_graph(dask_dataframe, directed=False, renumber=False):
    """
    dask_dataframe contains weighted and undirected edges with self
    loops. Multiple edges will likely be present as well.  The returned Graph
    object must be symmetrized and have self loops removed.
    """
    G = cugraph.Graph(directed=directed)
    G.from_dask_cudf_edgelist(dask_dataframe, source="src", destination="dst", renumber=renumber)
    return G


def benchmark_cugraph_graph_creation(scale, edgefactor, seed, directed, renumber):
    """
    Entry point for the benchmark.
    """
    dask_df = generate_edgelist(scale=scale, edgefactor=edgefactor, seed=seed, unweighted=True)
    dask_df = dask_df.astype("int64")
    dask_df = dask_df.reset_index(drop=True)
    #dask_df = dask_df.persist()
    #wait(dask_df)
    memory_est = dask_df.memory_usage().sum().compute()
    print(f"Edge List Memory = {sizeof_fmt(memory_est)}, Number of input edges  = {len(dask_df):,}")
    G = construct_graph(dask_df, directed=directed, renumber=renumber)
    print(f"Number of edges in final graph = {G.number_of_edges():,}")
    print("-" * 80)


def restart_client(client):
    """
    Restart the Dask client
    """
    Comms.destroy()
    client.restart()
    client = client.run(enable_spilling)
    Comms.initialize(p2p=True)

# call __main__ function
if __name__ == "__main__":
   client, cluster = start_dask_client(dask_worker_devices=[1], jit_unspill=False, device_memory_limit=1.2)
   enable_spilling()
   client.run(enable_spilling)
   for scale in [22,23,24,25]:
    for directed in [True, False]:
        for renumber in [True, False]:
            benchmark_cugraph_graph_creation(scale=scale, edgefactor=16,  seed=123, directed=directed, renumber=renumber)  
            restart_client(client)
            sleep(10)
        print("-"*40 + f"renumber completed" + "-"*40)
    print("-" * 40 + f"scale = {scale} completed" + "-" * 40)
   
   # Cleanup Dask Cluster
   stop_dask_client(client, cluster)
