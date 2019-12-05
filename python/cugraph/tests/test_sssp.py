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

import numpy as np
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


def cugraph_call(cu_M, source, edgevals=False):

    G = cugraph.DiGraph()
    if edgevals is True:
        G.from_cudf_edgelist(cu_M, source='0', target='1', edge_attr='2')
    else:
        G.from_cudf_edgelist(cu_M, source='0', target='1')
    print('sources size = ' + str(len(cu_M['0'])))
    print('destinations size = ' + str(len(cu_M['1'])))

    print('cugraph Solving... ')
    t1 = time.time()

    df = cugraph.sssp(G, source)

    t2 = time.time() - t1
    print('Cugraph Time : '+str(t2))

    if(np.issubdtype(df['distance'].dtype, np.integer)):
        max_val = np.iinfo(df['distance'].dtype).max
    else:
        max_val = np.finfo(df['distance'].dtype).max

    verts_np = df['vertex'].to_array()
    dist_np = df['distance'].to_array()
    pred_np = df['predecessor'].to_array()
    result = dict(zip(verts_np, zip(dist_np, pred_np)))
    return result, max_val


def networkx_call(M, source, edgevals=False):

    print('Format conversion ... ')
    M = M.tocsr()
    if M is None:
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')

    # Directed NetworkX graph
    Gnx = nx.DiGraph(M)

    print('NX Solving... ')
    t1 = time.time()

    if edgevals is False:
        path = nx.single_source_shortest_path_length(Gnx, source)
    else:
        path = nx.single_source_dijkstra_path_length(Gnx, source)

    t2 = time.time() - t1

    print('NX Time : ' + str(t2))

    return path, Gnx


DATASETS = ['../datasets/dolphins.csv',
            '../datasets/karate.csv',
            '../datasets/netscience.csv',
            '../datasets/email-Eu-core.csv']
SOURCES = [1]


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
@pytest.mark.parametrize('source', SOURCES)
def test_sssp(managed, pool, graph_file, source):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())
    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)
    cu_paths, max_val = cugraph_call(cu_M, source)
    nx_paths, Gnx = networkx_call(M, source)

    # Calculating mismatch
    err = 0
    for vid in cu_paths:
        # Validate vertices that are reachable
        # NOTE : If distance type is float64 then cu_paths[vid][0]
        # should be compared against np.finfo(np.float64).max)
        if (cu_paths[vid][0] != max_val):
            if(cu_paths[vid][0] != nx_paths[vid]):
                err = err + 1
            # check pred dist + 1 = current dist (since unweighted)
            pred = cu_paths[vid][1]
            if(vid != source and cu_paths[pred][0] + 1 != cu_paths[vid][0]):
                err = err + 1
        else:
            if (vid in nx_paths.keys()):
                err = err + 1

    assert err == 0


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', ['../datasets/netscience.csv'])
@pytest.mark.parametrize('source', SOURCES)
def test_sssp_edgevals(managed, pool, graph_file, source):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)
    cu_paths, max_val = cugraph_call(cu_M, source, edgevals=True)
    nx_paths, Gnx = networkx_call(M, source, edgevals=True)

    # Calculating mismatch
    err = 0
    for vid in cu_paths:
        # Validate vertices that are reachable
        # NOTE : If distance type is float64 then cu_paths[vid][0]
        # should be compared against np.finfo(np.float64).max)
        if (cu_paths[vid][0] != max_val):
            if(cu_paths[vid][0] != nx_paths[vid]):
                err = err + 1
            # check pred dist + edge_weight = current dist
            if(vid != source):
                pred = cu_paths[vid][1]
                edge_weight = Gnx[pred][vid]['weight']
                if(cu_paths[pred][0] + edge_weight != cu_paths[vid][0]):
                    err = err + 1
        else:
            if (vid in nx_paths.keys()):
                err = err + 1

    assert err == 0


@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', ['../datasets/netscience.csv'])
@pytest.mark.parametrize('source', SOURCES)
def test_sssp_data_type_conversion(managed, pool, graph_file, source):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)

    # cugraph call with int32 weights
    cu_M['2'] = cu_M['2'].astype(np.int32)
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source='0', target='1', edge_attr='2')
    # assert cugraph weights is int32
    assert G.edgelist.edgelist_df['weights'].dtype == np.int32
    df = cugraph.sssp(G, source)
    max_val = np.finfo(df['distance'].dtype).max
    verts_np = df['vertex'].to_array()
    dist_np = df['distance'].to_array()
    pred_np = df['predecessor'].to_array()
    cu_paths = dict(zip(verts_np, zip(dist_np, pred_np)))

    # networkx call with int32 weights
    M = M.tocsr()
    M.data = M.data.astype(np.int32)
    Gnx = nx.DiGraph(M)
    # assert nx weights is int32
    assert list(Gnx.edges(data=True))[0][2]['weight'].dtype == np.int32
    nx_paths = nx.single_source_dijkstra_path_length(Gnx, source)

    # Calculating mismatch
    err = 0
    for vid in cu_paths:
        # Validate vertices that are reachable
        # NOTE : If distance type is float64 then cu_paths[vid][0]
        # should be compared against np.finfo(np.float64).max)
        if (cu_paths[vid][0] != max_val):
            if(cu_paths[vid][0] != nx_paths[vid]):
                err = err + 1
            # check pred dist + edge_weight = current dist
            if(vid != source):
                pred = cu_paths[vid][1]
                edge_weight = Gnx[pred][vid]['weight']
                if(cu_paths[pred][0] + edge_weight != cu_paths[vid][0]):
                    err = err + 1
        else:
            if (vid in nx_paths.keys()):
                err = err + 1

    assert err == 0
