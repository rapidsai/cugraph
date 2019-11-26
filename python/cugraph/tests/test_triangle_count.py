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

import cudf
import cugraph
from cugraph.tests import utils
import rmm

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx


def cugraph_call(M, edgevals=False):
    M = M.tocoo()
    G = cugraph.DiGraph()
    cu_M = cudf.DataFrame()
    cu_M['src'] = cudf.Series(M.row)
    cu_M['dst'] = cudf.Series(M.col)
    if edgevals is True:
        cu_M['weights'] = cudf.Series(M.data)
        G.from_cudf_edgelist(cu_M, source='src', target='dst',
                             edge_attr='weights')
    else:
        G.from_cudf_edgelist(cu_M, source='src', target='dst')
    return cugraph.triangles(G)


def networkx_call(M):
    Gnx = nx.Graph(M)
    dic = nx.triangles(Gnx)
    count = 0
    for i in range(len(dic)):
        count += dic[i]
    return count


DATASETS = ['../datasets/dolphins.csv',
            '../datasets/karate.csv',
            '../datasets/netscience.csv']


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_triangles(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    M = utils.read_csv_for_nx(graph_file)
    cu_count = cugraph_call(M)
    nx_count = networkx_call(M)
    assert cu_count == nx_count


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_triangles_edge_vals(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    M = utils.read_csv_for_nx(graph_file)
    cu_count = cugraph_call(M, edgevals=True)
    nx_count = networkx_call(M)
    assert cu_count == nx_count
