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
import time

import pytest
import numpy as np
import scipy
import cugraph
from cugraph.tests import utils


def cugraph_call(cu_M, pairs, edgevals=False):
    G = cugraph.DiGraph()
    # Device data
    if edgevals is True:
        G.from_cudf_edgelist(cu_M, source='0', destination='1', edge_attr='2')
    else:
        G.from_cudf_edgelist(cu_M, source='0', destination='1')
    # cugraph Overlap Call
    t1 = time.time()
    df = cugraph.overlap(G, pairs)
    t2 = time.time() - t1
    print('Time : '+str(t2))
    df = df.sort_values(by=['source', 'destination'])
    return df['overlap_coeff'].to_array()


def intersection(a, b, M):
    count = 0
    a_idx = M.indptr[a]
    b_idx = M.indptr[b]

    while (a_idx < M.indptr[a+1]) and (b_idx < M.indptr[b+1]):
        a_vertex = M.indices[a_idx]
        b_vertex = M.indices[b_idx]

        if a_vertex == b_vertex:
            count += 1
            a_idx += 1
            b_idx += 1
        elif a_vertex < b_vertex:
            a_idx += 1
        else:
            b_idx += 1

    return count


def degree(a, M):
    return M.indptr[a+1] - M.indptr[a]


def overlap(a, b, M):
    b_sum = degree(b, M)
    if b_sum == 0:
        return float('NaN')

    a_sum = degree(a, M)

    i = intersection(a, b, M)
    total = min(a_sum, b_sum)
    return i / total


def cpu_call(M, first, second):
    result = []
    for i in range(len(first)):
        result.append(overlap(first[i], second[i], M))
    print(result)
    return result


# Test
@pytest.mark.parametrize('graph_file', utils.DATASETS)
def test_overlap(graph_file):
    gc.collect()

    Mnx = utils.read_csv_for_nx(graph_file)
    N = max(max(Mnx['0']), max(Mnx['1'])) + 1
    M = scipy.sparse.csr_matrix((Mnx.weight, (Mnx['0'], Mnx['1'])),
                                shape=(N, N))

    cu_M = utils.read_csv_file(graph_file)
    G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1')
    pairs = G.get_two_hop_neighbors()

    cu_coeff = cugraph_call(cu_M, pairs)
    cpu_coeff = cpu_call(M, pairs['first'], pairs['second'])

    assert len(cu_coeff) == len(cpu_coeff)
    for i in range(len(cu_coeff)):
        if np.isnan(cpu_coeff[i]):
            assert np.isnan(cu_coeff[i])
        elif np.isnan(cu_coeff[i]):
            assert cpu_coeff[i] == cu_coeff[i]
        else:
            diff = abs(cpu_coeff[i] - cu_coeff[i])
            assert diff < 1.0e-6


# Test
@pytest.mark.parametrize('graph_file', utils.DATASETS)
def test_overlap_edge_vals(graph_file):
    gc.collect()

    Mnx = utils.read_csv_for_nx(graph_file)
    N = max(max(Mnx['0']), max(Mnx['1'])) + 1
    M = scipy.sparse.csr_matrix((Mnx.weight, (Mnx['0'], Mnx['1'])),
                                shape=(N, N))

    cu_M = utils.read_csv_file(graph_file)
    G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1')
    pairs = G.get_two_hop_neighbors()

    cu_coeff = cugraph_call(cu_M, pairs,
                            edgevals=True)
    cpu_coeff = cpu_call(M, pairs['first'], pairs['second'])

    assert len(cu_coeff) == len(cpu_coeff)
    for i in range(len(cu_coeff)):
        if np.isnan(cpu_coeff[i]):
            assert np.isnan(cu_coeff[i])
        elif np.isnan(cu_coeff[i]):
            assert cpu_coeff[i] == cu_coeff[i]
        else:
            diff = abs(cpu_coeff[i] - cu_coeff[i])
            assert diff < 1.0e-6
