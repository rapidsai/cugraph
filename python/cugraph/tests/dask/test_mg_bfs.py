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


def test_dask_bfs():
    gc.collect()
    cluster = LocalCUDACluster()
    client = Client(cluster)
    Comms.initialize()

    #input_data_path = r"/home/aatish/workspace/datasets/GAP-road.csv"
    #input_data_path = r"/home/aatish/workspace/cugraph/datasets/karate.csv"
    #input_data_path = r"/home/aatish/workspace/cugraph/datasets/email-Eu-core.csv"
    input_data_path = r"/home/aatish/workspace/cugraph/datasets/netscience.csv"
    #input_data_path = r"/home/aatish/workspace/cugraph/datasets/hibench_small/1/part-00000.csv"
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
    g.from_cudf_edgelist(df, 'src', 'dst', renumber=True)

    dg = cugraph.DiGraph()
    dg.from_dask_cudf_edgelist(ddf, renumber=True)
    #warmup
    #result_dist = dcg.bfs(dg, 0, True)
    print("Warmup done")

    import time
    t1 = time.time()
    expected_dist = cugraph.bfs(g, 0)
    t2 = time.time()
    t3 = time.time()
    result_dist = dcg.bfs(dg, 0, True)
    t4 = time.time()
#    print("---------- MG BFS Call Time: ", t4-t3, "s ----------")
#    print("---------- SG BFS Call Time: ", t2-t1, "s ----------")
#
    compare_dist = expected_dist.merge(
        result_dist, on="vertex", suffixes=['_local', '_dask']
    )

    err = 0

    print("expected_dist len", len(expected_dist))
    print("result_dist len", len(result_dist))
    #assert len(expected_dist) == len(result_dist)
#    for i in range(len(compare_dist)):
#        err_str = ""
#        if (compare_dist['distance_local'].iloc[i] !=
#                compare_dist['distance_dask'].iloc[i]):
#            err_str = " e"
#            print(i, " ", compare_dist['vertex'].iloc[i], " ",
#                compare_dist['distance_local'].iloc[i], " ",
#                compare_dist['distance_dask'].iloc[i], err_str)
#    for i in range(len(compare_dist)):
#        if (compare_dist['distance_local'].iloc[i] !=
#                compare_dist['distance_dask'].iloc[i]):
#            err_str=" e"
#        else:
#            err_str=""
#        print(i, " ", compare_dist['vertex'].iloc[i],
#                " ", compare_dist['distance_local'].iloc[i], " ",
#                compare_dist['distance_dask'].iloc[i], err_str)
    for i in range(len(compare_dist)):
        if (compare_dist['distance_local'].iloc[i] !=
                compare_dist['distance_dask'].iloc[i]):
            print(i, " ", compare_dist['distance_local'].iloc[i], " ",
                    compare_dist['distance_dask'].iloc[i])
            err = err + 1
    assert err == 0

    Comms.destroy()
    client.close()
    cluster.close()
