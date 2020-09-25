# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
import cugraph.comms as Comms
from dask_cuda import LocalCUDACluster
import pytest


@pytest.fixture
def client_connection():
    cluster = LocalCUDACluster()
    client = Client(cluster)
    Comms.initialize()

    yield client

    Comms.destroy()
    client.close()
    cluster.close()


@pytest.mark.skip(reason="skipping MG testing on a SG system")
def test_compute_local_data(client_connection):

    gc.collect()

    input_data_path = r"../datasets/karate.csv"
    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize,
                             delimiter=' ',
                             names=['src', 'dst', 'value'],
                             dtype=['int32', 'int32', 'float32'])

    dg = cugraph.DiGraph()
    dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst',
                               edge_attr='value')

    # Compute_local_data
    dg.compute_local_data(by='dst')
    data = dg.local_data['data']
    by = dg.local_data['by']

    assert by == 'dst'
    assert Comms.is_initialized()

    global_num_edges = data.local_data['edges'].sum()
    assert global_num_edges == dg.number_of_edges()
    global_num_verts = data.local_data['verts'].sum()
    assert global_num_verts == dg.number_of_nodes()
