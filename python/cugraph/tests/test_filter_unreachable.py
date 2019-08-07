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

DATASETS = ['../datasets/dolphins',
            '../datasets/karate',
            '../datasets/netscience']

SOURCES = [1]

@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', ['../datasets/netscience'])
@pytest.mark.parametrize('source', SOURCES)
def test_filter_unreachable(managed, pool, graph_file, source):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm.initialize()

    assert(rmm.is_initialized())

    cu_M = utils.read_csv_file(graph_file+'.csv')
    # Device data
    sources = cu_M['0']
    destinations = cu_M['1']

    print('sources size = ' + str(len(sources)))
    print('destinations size = ' + str(len(destinations)))

    # cugraph Pagerank Call
    G = cugraph.Graph()
    G.add_edge_list(sources, destinations)

    print('cugraph Solving... ')
    t1 = time.time()

    df = cugraph.sssp(G, source)

    t2 = time.time() - t1
    print('Time : '+str(t2))

    reachable_df = cugraph.filter_unreachable(df)
