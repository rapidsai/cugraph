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

import numpy as np
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


def compare_edges(cg, nxg, verts):
    edgelist_df = cg.view_edge_list()
    assert cg.edgelist.weights is False
    assert len(edgelist_df) == nxg.size()
    for i in range(len(edgelist_df)):
        assert nxg.has_edge(verts[edgelist_df['src'][i]],
                            verts[edgelist_df['dst'][i]])
    return True


def cugraph_call(M, verts, directed=True):
    # directed is used to create either a Graph or DiGraph so the returned
    # cugraph can be compared to nx graph of same type.
    if directed:
        G = cugraph.DiGraph()
    else:
        G = cugraph.Graph()
    cu_M = cudf.DataFrame()
    cu_M['src'] = cudf.Series(M['0'])
    cu_M['dst'] = cudf.Series(M['1'])
    G.from_cudf_edgelist(cu_M, source='src', destination='dst')
    cu_verts = cudf.Series(verts)
    return cugraph.subgraph(G, cu_verts)


def nx_call(M, verts, directed=True):
    if directed:
        G = nx.from_pandas_edgelist(M, source='0', target='1',
                                    create_using=nx.DiGraph())
    else:
        G = nx.from_pandas_edgelist(M, source='0', target='1',
                                    create_using=nx.Graph())
    return nx.subgraph(G, verts)


DATASETS = ['../datasets/karate.csv',
            '../datasets/dolphins.csv',
            '../datasets/netscience.csv',
            '../datasets/email-Eu-core.csv']


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_subgraph_extraction_DiGraph(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    M = utils.read_csv_for_nx(graph_file)
    verts = np.zeros(3, dtype=np.int32)
    verts[0] = 0
    verts[1] = 1
    verts[2] = 17
    cu_sg = cugraph_call(M, verts)
    nx_sg = nx_call(M, verts)
    assert compare_edges(cu_sg, nx_sg, verts)


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_subgraph_extraction_Graph(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    M = utils.read_csv_for_nx(graph_file)
    verts = np.zeros(3, dtype=np.int32)
    verts[0] = 0
    verts[1] = 1
    verts[2] = 17
    cu_sg = cugraph_call(M, verts, False)
    nx_sg = nx_call(M, verts, False)
    assert compare_edges(cu_sg, nx_sg, verts)
