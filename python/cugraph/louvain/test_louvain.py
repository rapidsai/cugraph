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

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, these import community and import networkx need to be
# relocated in the third-party group once this gets fixed.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import community
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
    values = cudf.Series(M.data)

    G = cugraph.Graph()
    G.add_adj_list(row_offsets, col_indices, values)

    # cugraph Louvain Call
    t1 = time.time()
    parts, mod = cugraph.nvLouvain(G)
    t2 = time.time() - t1
    print('Time : '+str(t2))

    return parts, mod


def networkx_call(M):
    M = M.tocsr()

    # Directed NetworkX graph
    Gnx = nx.Graph(M)

    # z = {k: 1.0/M.shape[0] for k in range(M.shape[0])}

    # Networkx Jaccard Call
    print('Solving... ')
    t1 = time.time()
    parts = community.best_partition(Gnx)
    t2 = time.time() - t1

    print('Time : '+str(t2))
    return parts


DATASETS = ['/datasets/networks/karate.mtx',
            '/datasets/networks/dolphins.mtx',
            '/datasets/networks/netscience.mtx']


@pytest.mark.parametrize('graph_file', DATASETS)
def test_louvain(graph_file):
    M = read_mtx_file(graph_file)
    cu_parts, cu_mod = cugraph_call(M)
    nx_parts = networkx_call(M)

    # Calculating modularity scores for comparison
    Gnx = nx.Graph(M)
    cu_map = {0: 0}
    for i in range(len(cu_parts)):
        cu_map[cu_parts['vertex'][i]] = cu_parts['partition'][i]
    cu_mod_nx = community.modularity(cu_map, Gnx)
    nx_mod = community.modularity(nx_parts, Gnx)
    assert len(cu_parts) == len(nx_parts)
    assert cu_mod > (.82 * nx_mod)
    assert abs(cu_mod - cu_mod_nx) < .0001
