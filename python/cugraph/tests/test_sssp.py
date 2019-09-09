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


print('Networkx version : {} '.format(nx.__version__))


def cugraph_call(cu_M, source, edgevals=False):

    # Device data
    sources = cu_M['0']
    destinations = cu_M['1']
    if edgevals is False:
        values = None
    else:
        values = cu_M['2']

    print('sources size = ' + str(len(sources)))
    print('destinations size = ' + str(len(destinations)))

    # cugraph Pagerank Call
    G = cugraph.Graph()
    G.add_edge_list(sources, destinations, values)

    print('cugraph Solving... ')
    t1 = time.time()

    df = cugraph.sssp(G, source)

    t2 = time.time() - t1
    print('Time : '+str(t2))

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
    G = nx.Graph(M)
    Gnx = G.to_undirected()

    print('NX Solving... ')
    t1 = time.time()

    if edgevals is False:
        path = nx.single_source_shortest_path_length(Gnx, source)
    else:
        path = nx.single_source_dijkstra_path_length(Gnx, source)

    t2 = time.time() - t1

    print('Time : ' + str(t2))

    return path, Gnx


DATASETS = ['../datasets/dolphins',
            '../datasets/karate',
            '../datasets/netscience']
SOURCES = [1]


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
@pytest.mark.parametrize('source', SOURCES)
def test_sssp(managed, pool, graph_file, source):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm_cfg.initial_pool_size = 2 << 27
    rmm.initialize()

    assert(rmm.is_initialized())
    M = utils.read_mtx_file(graph_file+'.mtx')
    cu_M = utils.read_csv_file(graph_file+'.csv')
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
@pytest.mark.parametrize('graph_file', ['../datasets/netscience'])
@pytest.mark.parametrize('source', SOURCES)
def test_sssp_edgevals(managed, pool, graph_file, source):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm_cfg.initial_pool_size = 2 << 27
    rmm.initialize()

    assert(rmm.is_initialized())

    M = utils.read_mtx_file(graph_file+'.mtx')
    cu_M = utils.read_csv_file(graph_file+'.csv')
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

