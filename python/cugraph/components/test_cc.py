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
import queue
import time

import numpy as np
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


print('Networkx version : {} '.format(nx.__version__))


def read_mtx_file(mm_file):
    print('Reading ' + str(mm_file) + '...')
    return mmread(mm_file).asfptype()


def read_csv_file(mm_file):
    print('Reading ' + str(mm_file) + '...')
    return cudf.read_csv(mm_file, delimiter=' ',
                         dtype=['int32', 'int32', 'float32'], header=None)

def networkx_call(M):
    M = M.tocsr()
    if M is None:
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')

    Gnx = nx.DiGraph(M)

    # Weakly Connected components call:
    print('Solving... ')
    t1 = time.time()

    # same parameters as in NVGRAPH
    result = nx.weakly_connected_components(Gnx)
    t2 = time.time() - t1

    print('Time : ' + str(t2))

    labels = sorted(result)
    return labels


def cugraph_call(cu_M):
    # Device data
    sources = cu_M['0']
    destinations = cu_M['1']

    # cugraph Pagerank Call
    G = cugraph.Graph()
    G.add_edge_list(sources, destinations, None)
    t1 = time.time()
    df = cugraph.weak_cc(G)
    t2 = time.time() - t1
    print('Time : '+str(t2))

    result = df['labels'].to_array()

    labels = sorted(result)
    return labels

# these should come w/ cugraph/python:
#
DATASETS = ['../datasets/dolphins', '../datasets/karate'] #,
#            '../datasets/coPapersDBLP',     # missing
#            '../datasets/coPapersCiteseer', # missing
#            '../datasets/hollywood']        # missing


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_weak_cc(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm.initialize()

    assert(rmm.is_initialized())

    M = read_mtx_file(graph_file+'.mtx')
    netx_labels = networkx_call(M)
    
    cu_M = read_csv_file(graph_file+'.csv')
    cugraph_labels = cugraph_call(cu_M)

    # NetX returns a list of components, each component being a
    # collection (set{}) of vertex indices;
    #
    # while cugraph returns a component label for each vertex;

    nx_n_components = len(netx_labels)
    cg_n_components = max(cugraph_labels)
    
    assert nx_n_components == cg_n_components

    lst_nx_components_lens = [len(c) for c in sorted(netx_labels ,key=len)]

    # get counts of uniques:
    #
    counter_f = lambda ls, val: sum(1 for x in ls if x==val)
    lst_cg_components_lens = [counter_f(cugraph_labels, uniq_val) for uniq_val in set(cugraph_labels)]

    assert lst_nx_components_lens == lst_cg_components_lens
