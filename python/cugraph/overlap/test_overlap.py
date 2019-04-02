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

import time

import pytest
from scipy.io import mmread

import cudf
import cugraph

def read_mtx_file(mm_file):
    print('Reading ' + str(mm_file) + '...')
    return mmread(mm_file).asfptype()


def cugraph_call(M, first, second, edgevals=False):
    M = M.tocsr()
    if M is None:
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')

    # Device data
    row_offsets = cudf.Series(M.indptr)
    col_indices = cudf.Series(M.indices)
    if edgevals is False:
        values = None
    else:
        values = cudf.Series(M.data)

    G = cugraph.Graph()
    G.add_adj_list(row_offsets, col_indices, values)

    # cugraph Jaccard Call
    t1 = time.time()
    df = cugraph.overlap(G, first, second)
    t2 = time.time() - t1
    print('Time : '+str(t2))

    return df['overlap_coeff'].to_array()

def intersection(a, b, M):
    count = 0
    for idx in range(M.indptr[a], M.indptr[a+1]):
        a_node = M.indices[idx]
        for inner_idx in range(M.indptr[b], M.indptr[b+1]):
            b_node = M.indices[inner_idx]
            if a_node == b_node:
                count += 1
    return count

def degree(a, M):
    return M.indptr[a+1] - M.indptr[a]

def overlap(a, b, M):
    i = intersection(a, b, M)
    a_sum = degree(a, M)
    b_sum = degree(b, M)
    total = min(a_sum, b_sum)
    return i / total

def cpu_call(M, first, second):
    M = M.tocsr()
    result = []
    for i in range(len(first)):
        result.append(overlap(first[i], second[i], M))
    return result

DATASETS = ['/datasets/networks/dolphins.mtx',
            '/datasets/networks/karate.mtx',
            '/datasets/networks/netscience.mtx']


@pytest.mark.parametrize('graph_file', DATASETS)
def test_overlap(graph_file):
    M = read_mtx_file(graph_file)
    M = M.tocsr()
    row_offsets = cudf.Series(M.indptr)
    col_indices = cudf.Series(M.indices)
    G = cugraph.Graph()
    G.add_adj_list(row_offsets, col_indices, None)
    pairs = G.get_two_hop_neighbors()

    cu_coeff = cugraph_call(M, pairs['first'], pairs['second'])
    cpu_coeff = cpu_call(M, pairs['first'], pairs['second'])

    assert len(cu_coeff) == len(cpu_coeff)
    for i in range(len(cu_coeff)):
        diff = abs(cpu_coeff[i] - cu_coeff[i])
        assert diff < 1.0e-6
        
@pytest.mark.parametrize('graph_file', DATASETS)
def test_overlap_edge_vals(graph_file):
    M = read_mtx_file(graph_file)
    M = M.tocsr()
    row_offsets = cudf.Series(M.indptr)
    col_indices = cudf.Series(M.indices)
    G = cugraph.Graph()
    G.add_adj_list(row_offsets, col_indices, None)
    pairs = G.get_two_hop_neighbors()

    cu_coeff = cugraph_call(M, pairs['first'], pairs['second'], edgevals=True)
    cpu_coeff = cpu_call(M, pairs['first'], pairs['second'])

    assert len(cu_coeff) == len(cpu_coeff)
    for i in range(len(cu_coeff)):
        diff = abs(cpu_coeff[i] - cu_coeff[i])
        assert diff < 1.0e-6