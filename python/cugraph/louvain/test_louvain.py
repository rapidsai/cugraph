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
from scipy.io import mmread

import cudf
import cugraph
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg

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


def read_csv_file(mm_file):
    print('Reading ' + str(mm_file) + '...')
    return cudf.read_csv(mm_file, delimiter=' ',
                         dtype=['int32', 'int32', 'float32'], header=None)


def cugraph_call(cu_M, edgevals=False):

    # Device data
    sources = cu_M['0']
    destinations = cu_M['1']
    if edgevals:
        values = cu_M['2']
    else:
        values = None
    G = cugraph.Graph()
    G.add_edge_list(sources, destinations, values)

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

    # Networkx louvain Call
    print('Solving... ')
    t1 = time.time()
    parts = community.best_partition(Gnx)
    t2 = time.time() - t1

    print('Time : '+str(t2))
    return parts


DATASETS = ['../datasets/karate',
            '../datasets/dolphins',
            '../datasets/netscience']


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_louvain_with_edgevals(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm.initialize()

    assert(rmm.is_initialized())

    M = read_mtx_file(graph_file+'.mtx')
    cu_M = read_csv_file(graph_file+'.csv')
    cu_parts, cu_mod = cugraph_call(cu_M, edgevals=True)
    nx_parts = networkx_call(M)

    # Calculating modularity scores for comparison
    Gnx = nx.Graph(M)
    cu_map = {0: 0}
    for i in range(len(cu_parts)):
        cu_map[cu_parts['vertex'][i]] = cu_parts['partition'][i]
    assert set(nx_parts.keys()) == set(cu_map.keys())
    cu_mod_nx = community.modularity(cu_map, Gnx)
    nx_mod = community.modularity(nx_parts, Gnx)
    assert len(cu_parts) == len(nx_parts)
    assert cu_mod > (.82 * nx_mod)
    print(cu_mod)
    print(cu_mod_nx)
    print(nx_mod)
    assert abs(cu_mod - cu_mod_nx) < .0001


DATASETS = ['../datasets/karate',
            '../datasets/dolphins']


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_louvain(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm.initialize()

    assert(rmm.is_initialized())

    M = read_mtx_file(graph_file+'.mtx')
    cu_M = read_csv_file(graph_file+'.csv')
    cu_parts, cu_mod = cugraph_call(cu_M)
    nx_parts = networkx_call(M)

    # Calculating modularity scores for comparison
    Gnx = nx.Graph(M)
    cu_map = {0: 0}
    for i in range(len(cu_parts)):
        cu_map[cu_parts['vertex'][i]] = cu_parts['partition'][i]
    assert set(nx_parts.keys()) == set(cu_map.keys())
    cu_mod_nx = community.modularity(cu_map, Gnx)
    nx_mod = community.modularity(nx_parts, Gnx)
    assert len(cu_parts) == len(nx_parts)
    assert cu_mod > (.82 * nx_mod)
    print(cu_mod)
    print(cu_mod_nx)
    print(nx_mod)
    assert abs(cu_mod - cu_mod_nx) < .0001
