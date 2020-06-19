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
from dask.distributed import Client
import gc
import cugraph
import dask_cudf

# Move to conftest
from dask_cuda import LocalCUDACluster
# cluster = LocalCUDACluster(protocol="tcp", scheduler_port=0)
#


def test_dask_pagerank():

    gc.collect()
    cluster = LocalCUDACluster(protocol="tcp", scheduler_port=0)
    client = Client(cluster)

    input_data_path = r"../datasets/karate.csv"

    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize,
                             delimiter=' ',
                             names=['src', 'dst', 'value'],
                             dtype=['int32', 'int32', 'float32'])

    g = cugraph.DiGraph()
    g.from_dask_cudf_edgelist(ddf)

    dcg.pagerank(g)

    client.close()
    cluster.close()
