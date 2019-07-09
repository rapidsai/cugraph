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

import pytest
from scipy.io import mmread

import cudf
import cugraph
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx


def read_mtx_file(mm_file):
    print('Reading ' + str(mm_file) + '...')
    return mmread(mm_file).asfptype()


def cugraph_call(M, edgevals=False):
    M = M.tocoo()
    rows = cudf.Series(M.row)
    cols = cudf.Series(M.col)
    if edgevals is False:
        values = None
    else:
        values = cudf.Series(M.data)
    G = cugraph.Graph()
    G.add_edge_list(rows, cols, values)
    return cugraph.triangles(G)


def networkx_call(M):
    Gnx = nx.Graph(M)
    dic = nx.triangles(Gnx)
    count = 0
    for i in range(len(dic)):
        count += dic[i]
    return count


DATASETS = ['../datasets/dolphins.mtx',
            '../datasets/karate.mtx',
            '../datasets/netscience.mtx']


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_triangles(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm.initialize()

    assert(rmm.is_initialized())

    M = read_mtx_file(graph_file)
    cu_count = cugraph_call(M)
    nx_count = networkx_call(M)
    assert cu_count == nx_count


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_triangles_edge_vals(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm.initialize()

    assert(rmm.is_initialized())

    M = read_mtx_file(graph_file)
    cu_count = cugraph_call(M, edgevals=True)
    nx_count = networkx_call(M)
    assert cu_count == nx_count
