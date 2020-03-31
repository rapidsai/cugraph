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
import scipy
import cugraph
from cugraph.tests import utils
import rmm


def cugraph_call(cu_M, start_vertex):

    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1',
                         edge_attr='2')

    t1 = time.time()
    df = cugraph.bfs(G, start_vertex)
    t2 = time.time() - t1
    print('Time : '+str(t2))

    # Return distances as np.array()
    return df['vertex'].to_array(), df['distance'].to_array()


def base_call(M, start_vertex):
    int_max = 2**31 - 1
    N = max(max(M['0']), max(M['1'])) + 1
    M = scipy.sparse.csr_matrix((M.weight, (M['0'], M['1'])),
                                shape=(N, N))

    offsets = M.indptr
    indices = M.indices
    num_verts = len(offsets) - 1
    dist = np.zeros(num_verts, dtype=np.int32)
    vertex = list(range(num_verts))

    for i in range(num_verts):
        dist[i] = int_max

    q = queue.Queue()
    q.put(start_vertex)
    dist[start_vertex] = 0
    while(not q.empty()):
        u = q.get()
        for i_col in range(offsets[u], offsets[u + 1]):
            v = indices[i_col]
            if (dist[v] == int_max):
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
    cugraph_vid, cugraph_dist = cugraph_call(cu_M, 0)

    # Calculating mismatch
    # Currently, vertex order mismatch is not considered as an error
    cugraph_idx = 0
    base_idx = 0
    distance_error_counter = 0
    while cugraph_idx < len(cugraph_dist):
        if base_vid[base_idx] == cugraph_vid[cugraph_idx]:
            # An error is detected when for the same vertex
            # the distances are different
            if base_dist[base_idx] != cugraph_dist[cugraph_idx]:
                distance_error_counter += 1
            cugraph_idx += 1
        base_idx += 1
    assert distance_error_counter == 0
