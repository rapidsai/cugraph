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

import gc
from itertools import product
import queue
import time

import numpy as np
import pytest

import cugraph
from cugraph.tests import utils
import rmm

# compute once
_int_max = 2**31 - 1


def cugraph_call(cu_M, start_vertex):
    # Device data
    df = cu_M[['0', '1']]

    t1 = time.time()
    df = cugraph.bsp.traversal.bfs_df_pregel(
        df, start_vertex, src_col='0', dst_col='1')
    t2 = time.time() - t1
    print('Time : '+str(t2))

    # Return distances as np.array()
    return df['vertex'].to_array(), df['distance'].to_array()


def base_call(M, start_vertex):

    M = M.tocsr()

    offsets = M.indptr
    indices = M.indices
    num_verts = len(offsets) - 1
    dist = np.zeros(num_verts, dtype=np.int32)
    vertex = list(range(num_verts))

    for i in range(num_verts):
        dist[i] = _int_max

    q = queue.Queue()
    q.put(start_vertex)
    dist[start_vertex] = 0
    while(not q.empty()):
        u = q.get()
        for i_col in range(offsets[u], offsets[u + 1]):
            v = indices[i_col]
            if (dist[v] == _int_max):
                dist[v] = dist[u] + 1
                q.put(v)

    return vertex, dist


DATASETS = ['../datasets/dolphins.csv',
            '../datasets/karate.csv',
            '../datasets/polbooks.csv',
            '../datasets/netscience.csv',
            '../datasets/email-Eu-core.csv']

# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_bfs(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)

    base_vid, base_dist = base_call(M, 0)
    cugraph_vid, cugraph_dist = cugraph_call(cu_M, np.int32(0))

    # Calculating mismatch
    num_dist = np.count_nonzero(base_dist != _int_max)

    assert num_dist == len(cugraph_dist)
