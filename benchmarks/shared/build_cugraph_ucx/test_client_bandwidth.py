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
from dask.distributed import Client, wait
import cupy as cp
import numpy as np
import cudf
import dask_cudf
import rmm
from time import perf_counter_ns

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

def create_dataframe(client):
    n_rows = 25_000_000
    df = cudf.DataFrame({'src':cp.arange(0,n_rows,dtype=cp.int32), 'dst':cp.arange(0,n_rows, dtype=cp.int32), 'eids':cp.ones(n_rows, cp.int32)})
    ddf = dask_cudf.from_cudf(df,npartitions= len(client.scheduler_info()['workers'])).persist() 
    client.rebalance(ddf)
    del df
    _ = wait(ddf)
    return ddf

@benchmark_func
def get_n_rows(ddf, n):
    if n==-1:
        df = ddf.compute() 
    else:
        df = ddf.head(n)
    return df

def run_bandwidth_test(ddf, n):
    df, time_ls = get_n_rows(ddf, n)
    time_ar = np.asarray(time_ls)
    time_mean = time_ar.mean()
    size_bytes = df.memory_usage().sum()
    size_gb = round(size_bytes/(pow(1024,3)), 2)        
    print(f"Getting {len(df):,} rows  of size {size_gb} took = {time_mean*1e-6} ms")
    time_mean_s = time_mean*1e-9
    print(f"Bandwidth = {round(size_gb/time_mean_s, 4)} gb/s")
    return
    


if __name__ == "__main__":
    cluster = LocalCUDACluster(protocol='ucx',rmm_pool_size='15GB', CUDA_VISIBLE_DEVICES='1,2,3')
    client = Client(cluster)
    rmm.reinitialize(pool_allocator=True)

    ddf = create_dataframe(client)
    run_bandwidth_test(ddf, 1_000_000)
    run_bandwidth_test(ddf, 2_000_000)
    run_bandwidth_test(ddf, 4_000_000)
    run_bandwidth_test(ddf, -1)

    print("--"*20+"Completed Test"+"--"*20, flush=True)
    client.shutdown()
    cluster.close()

