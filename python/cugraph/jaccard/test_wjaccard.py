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
import numpy as np
import pytest
from scipy.io import mmread

import cudf
import cugraph

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx

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
    weights_arr = cudf.Series(np.ones(len(row_offsets) - 1, dtype=np.float32))

    G = cugraph.Graph()
    G.add_adj_list(row_offsets, col_indices, None)

    # cugraph Jaccard Call
    t1 = time.time()
    df = cugraph.jaccard_w(G, weights_arr)
    t2 = time.time() - t1
    print('Time : '+str(t2))

    return df['jaccard_coeff']


def networkx_call(M):

    M = M.tocsr()
    M = M.tocoo()
    sources = M.row
    destinations = M.col
    edges = []
    for i in range(len(sources)):
        edges.append((sources[i], destinations[i]))
    # in NVGRAPH tests we read as CSR and feed as CSC, so here we doing this
    # explicitly
    print('Format conversion ... ')

    # Directed NetworkX graph
    G = nx.DiGraph(M)
    Gnx = G.to_undirected()

    # Networkx Jaccard Call
    print('Solving... ')
    t1 = time.time()
    preds = nx.jaccard_coefficient(Gnx, edges)
    t2 = time.time() - t1

    print('Time : '+str(t2))
    coeff = []
    for u, v, p in preds:
        coeff.append(p)
    return coeff


DATASETS = ['../datasets/dolphins.mtx',
            '../datasets/karate.mtx',
            '../datasets/netscience.mtx']


@pytest.mark.parametrize('graph_file', DATASETS)
def test_wjaccard(graph_file):

    M = read_mtx_file(graph_file)
    # suppress F841 (local variable is assigned but never used) in flake8
    # no networkX equivalent to compare cu_coeff against...
    cu_coeff = cugraph_call(M)  # noqa: F841
    nx_coeff = networkx_call(M)
    for i in range(len(cu_coeff)):
        diff = abs(nx_coeff[i] - cu_coeff[i])
        assert diff < 1.0e-6
