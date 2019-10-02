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


def calc_k_cores(graph_file):
    M = utils.read_csv_file(graph_file + ".csv")
    G = cugraph.Graph()
    G.add_edge_list(M['0'], M['1'])

    ck = cugraph.k_core(G)

    NM = utils.read_mtx_file(graph_file + ".mtx")
    NM = NM.tocsr()
    Gnx = nx.DiGraph(NM)
    nk = nx.k_core(Gnx)
    return ck, nk


def compare_edges(cg, nxg):
    src, dest, weight = cg.view_edge_list()
    assert weight is None
    assert len(src) == nxg.size()
    for i in range(len(src)):
        assert nxg.has_edge(src[i], dest[i])
    return True


DATASETS = ['../datasets/dolphins', '../datasets/netscience']


@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_core_number(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm.initialize()

    assert(rmm.is_initialized())

    cu_kcore, nx_kcore = calc_k_cores(graph_file)

    assert compare_edges(cu_kcore, nx_kcore)
