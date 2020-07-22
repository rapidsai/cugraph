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
import numpy as np
import pytest
import cugraph.dask as dcg
import cugraph.comms as Comms
from dask.distributed import Client
import gc
import cugraph
import dask_cudf
import cudf
from dask_cuda import LocalCUDACluster

#The function selects personalization_perc% of accessible vertices in graph M
#and randomly assigns them personalization values
def personalize(v, personalization_perc)
    personalization = None
    if personalization_perc != 0:
        personalization = {}
        nnz_vtx = np.arrange(0,v)
        print(nnz_vtx)
        personalization_count = int((nnz_vtx.size *
                                     personalization_perc)/100.0)
        print(personalization_count)
        nnz_vtx = np.random.choice(nnz_vtx,
                                   min(nnz_vtx.size, personalization_count),
                                   replace=False)
        print(nnz_vtx)
        nnz_val = np.random.random(nnz_vtx.size)
        nnz_val = nnz_val/sum(nnz_val)
        print(nnz_val)
        for vtx, val in zip(nnz_vtx, nnz_val):
            personalization[vtx] = val
        cu_personalization = cudify(personalization)
    return cu_personalization

PERSONALIZATION_PERC = [0, 10, 50]
@pytest.mark.parametrize('personalization_perc', PERSONALIZATION_PERC)

def test_dask_pagerank(personalization_perc):
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

    # Pre compute local data and personalize
    if personalization_perc != 0:
        dg.compute_local_data(by='dst')
        test_dask_pagerank(dg.number_of_vertices(), personalization_perc)

    expected_pr = cugraph.pagerank(g, personalization=personalization, tol=1e-6)
    result_pr = dcg.pagerank(dg, personalization=personalization, tol=1e-6)

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