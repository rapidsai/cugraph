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
import time

import pytest

import cudf
import cugraph
from cugraph.tests import utils
import rmm
import numpy as np


def cugraph_call(cu_M, first, second):
    # Device data
    weights_arr = cudf.Series(np.ones(max(cu_M['0'].max(),
                              cu_M['1'].max())+1, dtype=np.float32))

    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source='0', target='1')

    # cugraph Overlap Call
    t1 = time.time()
    df = cugraph.overlap_w(G, weights_arr, first, second)
    t2 = time.time() - t1
    print('Time : '+str(t2))
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

    i = intersection(a, b, M)
    a_sum = degree(a, M)
    total = min(a_sum, b_sum)
    return i / total


def cpu_call(M, first, second):
    result = []
    for i in range(len(first)):
        result.append(overlap(first[i], second[i], M))
    return result


DATASETS = ['../datasets/dolphins.csv',
            '../datasets/karate.csv',
            '../datasets/netscience.csv']
#  Too slow to run on CPU
#            '../datasets/email-Eu-core.csv']


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_woverlap(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    M = utils.read_csv_for_nx(graph_file)
    M = M.tocsr().sorted_indices()
    cu_M = utils.read_csv_file(graph_file)
    row_offsets = cudf.Series(M.indptr)
    col_indices = cudf.Series(M.indices)
    G = cugraph.Graph()
    G.from_cudf_adjlist(row_offsets, col_indices, None)
    pairs = G.get_two_hop_neighbors()

    cu_coeff = cugraph_call(cu_M, pairs['first'], pairs['second'])
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
