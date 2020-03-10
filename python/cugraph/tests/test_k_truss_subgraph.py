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


def ktruss_ground_truth(graph_file):
    subgraph = utils.read_csv_for_nx(graph_file)
    nxktruss_subgraph = nx.DiGraph(subgraph.tocsr())
    return nxktruss_subgraph


def cugraph_k_truss_subgraph(graph_file, k):
    cu_M = utils.read_csv_file(graph_file)
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1')
    k_subgraph = cugraph.ktruss_subgraph(G, k)
    return k_subgraph


def compare_k_truss(graph_file, k, ground_truth_file):
    k_truss_cugraph = cugraph_k_truss_subgraph(graph_file, k)
    k_truss_nx = ktruss_ground_truth(ground_truth_file)

    edgelist_df = k_truss_cugraph.view_edge_list()
    src, dest = edgelist_df['src'], edgelist_df['dst'],
    for i in range(len(src)):
        assert (k_truss_nx.has_edge(src[i], dest[i]) or
                k_truss_nx.has_edge(dest[i], src[i]))
    return True


DATASETS = [('../datasets/polbooks.csv',
             '../datasets/ref/ktruss/polbooks.csv'),
            ('../datasets/netscience.csv',
             '../datasets/ref/ktruss/netscience.csv')]


@pytest.mark.parametrize('managed, pool',
                         list(product([False], [False])))
@pytest.mark.parametrize('graph_file, nx_ground_truth', DATASETS)
def test_ktruss_subgraph(managed, pool, graph_file, nx_ground_truth):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool)

    assert(rmm.is_initialized())

    compare_k_truss(graph_file, 5, nx_ground_truth)
