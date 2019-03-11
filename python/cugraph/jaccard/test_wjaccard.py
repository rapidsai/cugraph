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

import networkx as nx
import numpy as np
import pytest
from scipy.io import mmread

import cudf
import cugraph


print('Networkx version : {} '.format(nx.__version__))


def read_mtx_file(mm_file):
    print('Reading ' + str(mm_file) + '...')
    return mmread(mm_file).asfptype()


def cugraph_call(M):
    M = M.tocsr()
    if M is None:
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')

    # Device data
    row_offsets = cudf.Series(M.indptr)
    col_indices = cudf.Series(M.indices)
    # values = cudf.Series(np.ones(len(col_indices), dtype=np.float32),
    # nan_as_null=False)
    weights_arr = cudf.Series(np.ones(len(row_offsets), dtype=np.float32),
                              nan_as_null=False)

    G = cugraph.Graph()
    G.add_adj_list(row_offsets, col_indices, None)

    # cugraph Jaccard Call
    t1 = time.time()
    df = cugraph.nvJaccard_w(G, weights_arr)
    t2 = time.time() - t1
    print('Time : '+str(t2))

    return df['jaccard_coeff']

def cugraph_edge_call(M):
    M = M.tocoo()
    if M is None:
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')

    # Device data
    row = cudf.Series(M.row)
    col = cudf.Series(M.col)
    # values = cudf.Series(np.ones(len(col_indices), dtype=np.float32),
    # nan_as_null=False)
    

    G = cugraph.Graph()
    G.add_edge_list(row, col, None)
    
    weights_arr = cudf.Series(np.ones(G.num_vertices(), dtype=np.float32),
                              nan_as_null=False)

    # cugraph Jaccard Call
    t1 = time.time()
    df = cugraph.nvJaccard_w(G, weights_arr)
    t2 = time.time() - t1
    print('Time : '+str(t2))

    return df['jaccard_coeff']


DATASETS = ['/datasets/networks/dolphins.mtx',
            '/datasets/networks/karate.mtx',
            '/datasets/golden_data/graphs/dblp.mtx']


@pytest.mark.parametrize('graph_file', DATASETS)
def test_wjaccard_adjacency(graph_file):

    M = read_mtx_file(graph_file)
    # suppress F841 (local variable is assigned but never used) in flake8
    # no networkX equivalent to compare cu_coeff against...
    cu_coeff = cugraph_call(M)  # noqa: F841
    # this test is incomplete...
    
@pytest.mark.parametrize('graph_file', DATASETS)
def test_wjaccard_edge(graph_file):

    M = read_mtx_file(graph_file)
    # suppress F841 (local variable is assigned but never used) in flake8
    # no networkX equivalent to compare cu_coeff against...
    cu_coeff = cugraph_edge_call(M)  # noqa: F841
    # this test is incomplete...
