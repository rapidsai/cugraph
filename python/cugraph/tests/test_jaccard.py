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


print('Networkx version : {} '.format(nx.__version__))


def cugraph_call(cu_M, edgevals=False):
    G = cugraph.DiGraph()
    if edgevals is True:
        G.from_cudf_edgelist(cu_M, source='0', destination='1',
                             edge_attr='2')
    else:
        G.from_cudf_edgelist(cu_M, source='0', destination='1')

    # cugraph Jaccard Call
    t1 = time.time()
    df = cugraph.jaccard(G)
    t2 = time.time() - t1
    print('Time : '+str(t2))
    print(df)
    return df['source'].to_array(), df['destination'].to_array(),\
        df['jaccard_coeff'].to_array()


def networkx_call(M):

    sources = M['0']
    destinations = M['1']
    edges = []
    for i in range(len(M)):
        edges.append((sources[i], destinations[i]))
    edges = sorted(edges)
    # in NVGRAPH tests we read as CSR and feed as CSC, so here we doing this
    # explicitly
    print('Format conversion ... ')

    Gnx = nx.from_pandas_edgelist(M, source='0', target='1',
                                  edge_attr='weight', create_using=nx.Graph())
    # Networkx Jaccard Call
    print('Solving... ')
    t1 = time.time()
    preds = nx.jaccard_coefficient(Gnx, edges)
    t2 = time.time() - t1

    print('Time : '+str(t2))
    src = []
    dst = []
    coeff = []
    for u, v, p in preds:
        src.append(u)
        dst.append(v)
        coeff.append(p)
    return src, dst, coeff


DATASETS = ['../datasets/dolphins.csv',
            '../datasets/karate.csv',
            '../datasets/netscience.csv']


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_jaccard(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)
    cu_src, cu_dst, cu_coeff = cugraph_call(cu_M)
    nx_src, nx_dst, nx_coeff = networkx_call(M)

    # Calculating mismatch
    err = 0
    tol = 1.0e-06

    assert len(cu_coeff) == len(nx_coeff)
    for i in range(len(cu_coeff)):
        if(abs(cu_coeff[i] - nx_coeff[i]) > tol*1.1):
            err += 1

    print("Mismatches:  %d" % err)
    assert err == 0


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', ['../datasets/netscience.csv'])
def test_jaccard_edgevals(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)
    cu_src, cu_dst, cu_coeff = cugraph_call(cu_M, edgevals=True)
    nx_src, nx_dst, nx_coeff = networkx_call(M)

    # Calculating mismatch
    err = 0
    tol = 1.0e-06

    assert len(cu_coeff) == len(nx_coeff)
    for i in range(len(cu_coeff)):
        if(abs(cu_coeff[i] - nx_coeff[i]) > tol*1.1):
            err += 1

    print("Mismatches:  %d" % err)
    assert err == 0


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_jaccard_two_hop(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)

    Gnx = nx.from_pandas_edgelist(M, source='0', target='1',
                                  create_using=nx.Graph())
    G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1')
    pairs = G.get_two_hop_neighbors()
    nx_pairs = []
    for i in range(len(pairs)):
        nx_pairs.append((pairs['first'][i], pairs['second'][i]))
    preds = nx.jaccard_coefficient(Gnx, nx_pairs)
    nx_coeff = []
    for u, v, p in preds:
        nx_coeff.append(p)
    df = cugraph.jaccard(G, pairs)
    df = df.sort_values(by=['source', 'destination'])
    assert len(nx_coeff) == len(df)
    for i in range(len(df)):
        diff = abs(nx_coeff[i] - df['jaccard_coeff'][i])
        assert diff < 1.0e-6


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_jaccard_two_hop_edge_vals(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)

    Gnx = nx.from_pandas_edgelist(M, source='0', target='1',
                                  edge_attr='weight', create_using=nx.Graph())
    G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1', edge_attr='2')
    pairs = G.get_two_hop_neighbors()
    nx_pairs = []
    for i in range(len(pairs)):
        nx_pairs.append((pairs['first'][i], pairs['second'][i]))
    preds = nx.jaccard_coefficient(Gnx, nx_pairs)
    nx_coeff = []
    for u, v, p in preds:
        nx_coeff.append(p)
    df = cugraph.jaccard(G, pairs)
    df = df.sort_values(by=['source', 'destination'])
    assert len(nx_coeff) == len(df)
    for i in range(len(df)):
        diff = abs(nx_coeff[i] - df['jaccard_coeff'][i])
        assert diff < 1.0e-6
