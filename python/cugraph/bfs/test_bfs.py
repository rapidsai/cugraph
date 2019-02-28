# Copyright (c) 2019, NVIDIA CORPORATION.
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

import pytest
import queue
import time
import numpy as np
from scipy.io import mmread
import cudf
import cugraph


def ReadMtxFile(mmFile):
    print('Reading ' + str(mmFile) + '...')
    return mmread(mmFile).asfptype()


def cugraph_Call(M, start_vertex):
    # Device data
    M = M.tocsr()
    sources = cudf.Series(M.indptr)
    destinations = cudf.Series(M.indices)
    values = cudf.Series(M.data)

    G = cugraph.Graph()
    G.add_adj_list(sources, destinations, values)

    t1 = time.time()
    df = cugraph.bfs(G, start_vertex)
    t2 = time.time() - t1
    print('Time : '+str(t2))

    # Return distances as np.array()
    return df['distance'].to_array()


def base_Call(M, start_vertex):
    int_max = 2**31 - 1

    M = M.tocsr()

    offsets = M.indptr
    indices = M.indices
    num_verts = len(offsets) - 1
    dist = np.zeros(num_verts, dtype=np.int32)

    for i in range(num_verts):
        dist[i] = int_max

    q = queue.Queue()
    q.put(start_vertex)
    dist[start_vertex] = 0
    while(not q.empty()):
        u = q.get()
        for iCol in range(offsets[u], offsets[u + 1]):
            v = indices[iCol]
            if (dist[v] == int_max):
                dist[v] = dist[u] + 1
                q.put(v)

    return dist


datasets = ['/datasets/networks/dolphins.mtx',
            '/datasets/networks/karate.mtx',
            '/datasets/networks/polbooks.mtx',
            '/datasets/golden_data/graphs/dblp.mtx']


@pytest.mark.parametrize('graph_file', datasets)
def test_bfs(graph_file):
    M = ReadMtxFile(graph_file)

    base_dist = base_Call(M, 0)
    cugraph_dist = cugraph_Call(M, 0)

    # Calculating mismatch

    assert len(base_dist) == len(cugraph_dist)
    for i in range(len(cugraph_dist)):
        assert base_dist[i] == cugraph_dist[i]
