# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import rmm
import numpy as np
from time import perf_counter_ns
from cugraph.dask.comms import comms as Comms
from cugraph.dask import uniform_neighbor_sample as uniform_neighbor_sample_mg
from cugraph import MultiGraph
from cugraph.generators import rmat
import cudf

_seed = 42

def benchmark_func(func, n_times=10):
    def wrap_func(*args, **kwargs):
        time_ls = []
        # ignore 1st run
        # and return other runs
        for _ in range(0,n_times+1):
            t1 = perf_counter_ns()
            result = func(*args, **kwargs)
            t2 = perf_counter_ns()
            time_ls.append(t2-t1)
        return result, time_ls[1:]
    return wrap_func

def create_edgelist_df(graph_data):
    """
    Create a graph instance based on the data to be loaded/generated.
    """
  
    # Assume strings are names of datasets in the datasets package
    scale = graph_data["scale"]
    num_edges = (2**scale) * graph_data["edgefactor"]
    seed = _seed
    edgelist_df = rmat(
        scale,
        num_edges,
        0.57,  # from Graph500
        0.19,  # from Graph500
        0.19,  # from Graph500
        seed,
        clip_and_flip=False,
        scramble_vertex_ids=False,  # FIXME: need to understand relevance of this
        create_using=None,  # None == return edgelist
        mg=True,
    )
    edgelist_df["weight"] = np.float32(1)
    return edgelist_df
    
def create_mg_graph(edgelist_df):
    G = MultiGraph(directed=True)
    G.from_dask_cudf_edgelist(
        edgelist_df,
        source="src",
        destination="dst",
        edge_attr="weight",
        legacy_renum_only=True,
    )
    
    G.input_df = G.input_df.to_dask_dataframe()
    G.edgelist.edgelist_df = G.edgelist.edgelist_df.to_dask_dataframe()
    return G

@benchmark_func
def sample_graph(G, start_list):
    output_ddf = uniform_neighbor_sample_mg(G,start_list=start_list, fanout_vals=[10,25])
    df = output_ddf.compute()
    return df

def run_sampling_test(G, start_list):
    df, time_ls = sample_graph(G, start_list)
    time_ar = np.asarray(time_ls)
    time_mean = time_ar.mean()
    print(f"Sampling {len(start_list):,} took = {time_mean*1e-6} ms", flush=True)
    return


def start_cluster(n_workers):
    dask_worker_devices = ','.join([str(i) for i in range(1,n_workers+1)])    
    cluster = LocalCUDACluster(protocol='ucx',rmm_pool_size='25GB', CUDA_VISIBLE_DEVICES=dask_worker_devices)
    client = Client(cluster)
    Comms.initialize(p2p=True)
    rmm.reinitialize(pool_allocator=True, initial_pool_size=2**30, maximum_pool_size=2*(2**30) )
    return cluster, client



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", default=2, type=int)
    parser.add_argument("--scale", default=25, type=int)
    parser.add_argument("--edgefactor",default=16, type=int)
    args = parser.parse_args()
    print(args)
    cluster, client = start_cluster(args.n_workers)
    graph_data = {"scale": args.scale,
              "edgefactor": args.edgefactor,
              }    
    edgelist_df = create_edgelist_df(graph_data)
    g = create_mg_graph(edgelist_df)
    del edgelist_df

    for num_start_verts in [100, 500, 1_000, 5_000]+[i for i in range(10_000, 110_000, 10_000)]:
        start_list = g.input_df["src"].head(num_start_verts)
        start_list = cudf.Series(start_list)
        assert len(start_list)==num_start_verts
        run_sampling_test(g, start_list)
    
    print("--"*20+"Completed Test"+"--"*20, flush=True)

    Comms.destroy()
    client.shutdown()
    cluster.close()