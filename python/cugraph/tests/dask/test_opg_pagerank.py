# Copyright (c) 2020, NVIDIA CORPORATION.
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

import cugraph.dask as dcg
import cugraph.comms as Comms
from dask.distributed import Client
import gc
import cugraph
import dask_cudf
import cudf
from dask_cuda import LocalCUDACluster


def test_dask_pagerank():
    gc.collect()
    cluster = LocalCUDACluster()
    client = Client(cluster)
    Comms.initialize()

    input_data_path = r"../datasets/karate.csv"
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize,
                             delimiter=' ',
                             names=['src', 'dst', 'value'],
                             dtype=['int32', 'int32', 'float32'])

    df = cudf.read_csv(input_data_path,
                       delimiter=' ',
                       names=['src', 'dst', 'value'],
                       dtype=['int32', 'int32', 'float32'])

    g = cugraph.DiGraph()
    g.from_cudf_edgelist(df, 'src', 'dst')

    dg = cugraph.DiGraph()
    dg.from_dask_cudf_edgelist(ddf)

    # Pre compute local data
    # dg.compute_local_data(by='dst')

    expected_pr = cugraph.pagerank(g)
    result_pr = dcg.pagerank(dg, tol=1e-6)

    err = 0
    tol = 1.0e-05

    assert len(expected_pr) == len(result_pr)
    for i in range(len(result_pr)):
        if(abs(result_pr['pagerank'].iloc[i]-expected_pr['pagerank'].iloc[i])
           > tol*1.1):
            err = err + 1
    print("Mismatches:", err)
    assert err == 0

    Comms.destroy()
    client.close()
    cluster.close()
