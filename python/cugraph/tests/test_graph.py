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
import pandas as pd
import pytest

import cudf
import cudf._lib as libcudf
import cugraph
from cugraph.tests import utils
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg
'''
import socket
import struct
'''

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx


def compare_series(series_1, series_2):
    if (len(series_1) != len(series_2)):
        print("Series do not match in length")
        return 0
    for i in range(len(series_1)):
        if(series_1[i] != series_2[i]):
            print("Series[" + str(i) + "] does not match, " + str(series_1[i])
                  + ", " + str(series_2[i]))
            return 0
    return True


def compare_offsets(offset0, offset1):
    if not (len(offset0) <= len(offset1)):
        print("Mismatched length: " + str(len(offset0)) + " != "
              + str(len(offset1)))
        return False
    for i in range(len(offset0)):
        if offset0[i] != offset1[i]:
            print("Series[" + str(i) + "]: " + str(offset0[i]) + " != "
                  + str(offset1[i]))
            return False
    return True


# This function returns True if two graphs are identical (bijection between the
# vertices in one graph to the vertices in the other graph is identity AND two
# graphs are automorphic; no permutations of vertices are allowed).
def compare_graphs(nx_graph, cu_graph):
    sources, destinations, values = cu_graph.view_edge_list()

    df = cudf.DataFrame()
    df['source'] = sources
    df['target'] = destinations
    if values is not None:
        df['weight'] = values
        cu_to_nx_graph = nx.from_pandas_edgelist(df.to_pandas(),
                                                 source='source',
                                                 target='target',
                                                 edge_attr=['weight'],
                                                 create_using=nx.DiGraph())
    else:
        cu_to_nx_graph = nx.from_pandas_edgelist(df.to_pandas(),
                                                 create_using=nx.DiGraph())

    # first compare nodes

    ds0 = pd.Series(nx_graph.nodes)
    ds1 = pd.Series(cu_to_nx_graph.nodes)

    if not ds0.equals(ds1):
        return False

    # second compare edges

    diff = nx.difference(nx_graph, cu_to_nx_graph)
    if diff.number_of_edges() > 0:
        return False

    diff = nx.difference(cu_to_nx_graph, nx_graph)
    if diff.number_of_edges() > 0:
        return False

    if values is not None:
        df0 = cudf.from_pandas(nx.to_pandas_edgelist(nx_graph))
        df0 = df0.sort_values(by=['source', 'target'])
        df1 = df.sort_values(by=['source', 'target'])
        if not df0['weight'].equals(df1['weight']):
            return False

    return True


def find_two_paths(df, M):
    for i in range(len(df)):
        start = df['first'][i]
        end = df['second'][i]
        foundPath = False
        for idx in range(M.indptr[start], M.indptr[start + 1]):
            mid = M.indices[idx]
            for innerIdx in range(M.indptr[mid], M.indptr[mid + 1]):
                if M.indices[innerIdx] == end:
                    foundPath = True
                    break
            if foundPath:
                break
        if not foundPath:
            print("No path found between " + str(start) +
                  " and " + str(end))
        assert foundPath


def has_pair(first_arr, second_arr, first, second):
    for i in range(len(first_arr)):
        firstMatch = first_arr[i] == first
        secondMatch = second_arr[i] == second
        if firstMatch and secondMatch:
            return True
    return False


def check_all_two_hops(df, M):
    num_verts = len(M.indptr) - 1
    first_arr = df['first'].to_array()
    second_arr = df['second'].to_array()
    for start in range(num_verts):
        for idx in range(M.indptr[start], M.indptr[start + 1]):
            mid = M.indices[idx]
            for innerIdx in range(M.indptr[mid], M.indptr[mid + 1]):
                end = M.indices[innerIdx]
                if start != end:
                    assert has_pair(first_arr, second_arr, start, end)


def test_version():
    gc.collect()
    cugraph.__version__


DATASETS = ['../datasets/karate',
            '../datasets/dolphins',
            '../datasets/netscience']


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_add_edge_list_to_adj_list(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm_cfg.initial_pool_size = 2 << 27
    rmm.initialize()

    assert(rmm.is_initialized())

    cu_M = utils.read_csv_file(graph_file+'.csv')
    sources = cu_M['0']
    destinations = cu_M['1']

    M = utils.read_mtx_file(graph_file+'.mtx').tocsr()
    if M is None:
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')

    offsets_exp = M.indptr
    indices_exp = M.indices

    # cugraph add_egde_list to_adj_list call
    G = cugraph.Graph()
    G.add_edge_list(sources, destinations, None)
    offsets_cu, indices_cu, values_cu = G.view_adj_list()
    assert compare_offsets(offsets_cu, offsets_exp)
    assert compare_series(indices_cu, indices_exp)
    assert values_cu is None


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_add_adj_list_to_edge_list(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm_cfg.initial_pool_size = 2 << 27
    rmm.initialize()

    assert(rmm.is_initialized())

    M = utils.read_mtx_file(graph_file+'.mtx').tocsr()
    if M is None:
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')

    offsets = cudf.Series(M.indptr)
    indices = cudf.Series(M.indices)

    M = M.tocoo()
    sources_exp = cudf.Series(M.row)
    destinations_exp = cudf.Series(M.col)

    # cugraph add_adj_list to_edge_list call
    G = cugraph.Graph()
    G.add_adj_list(offsets, indices, None)
    sources, destinations, values = G.view_edge_list()
    sources_cu = np.array(sources)
    destinations_cu = np.array(destinations)
    assert compare_series(sources_cu, sources_exp)
    assert compare_series(destinations_cu, destinations_exp)
    assert values is None


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_transpose_from_adj_list(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm_cfg.initial_pool_size = 2 << 27
    rmm.initialize()

    assert(rmm.is_initialized())

    M = utils.read_mtx_file(graph_file+'.mtx').tocsr()
    offsets = cudf.Series(M.indptr)
    indices = cudf.Series(M.indices)
    G = cugraph.Graph()
    G.add_adj_list(offsets, indices, None)
    G.add_transposed_adj_list()
    Mt = M.transpose().tocsr()
    toff, tind, tval = G.view_transposed_adj_list()
    assert compare_series(tind, Mt.indices)
    assert compare_offsets(toff, Mt.indptr)
    assert tval is None


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_view_edge_list_from_adj_list(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm_cfg.initial_pool_size = 2 << 27
    rmm.initialize()

    assert(rmm.is_initialized())

    M = utils.read_mtx_file(graph_file+'.mtx').tocsr()
    offsets = cudf.Series(M.indptr)
    indices = cudf.Series(M.indices)
    G = cugraph.Graph()
    G.add_adj_list(offsets, indices, None)
    src2, dst2, val2 = G.view_edge_list()
    M = M.tocoo()
    src1 = M.row
    dst1 = M.col
    assert compare_series(src1, src2)
    assert compare_series(dst1, dst2)
    assert val2 is None


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_delete_edge_list_delete_adj_list(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm_cfg.initial_pool_size = 2 << 27
    rmm.initialize()

    assert(rmm.is_initialized())

    M = utils.read_mtx_file(graph_file+'.mtx')
    sources = cudf.Series(M.row)
    destinations = cudf.Series(M.col)

    M = M.tocsr()
    if M is None:
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')

    offsets = cudf.Series(M.indptr)
    indices = cudf.Series(M.indices)

    # cugraph delete_adj_list delete_edge_list call
    G = cugraph.Graph()
    G.add_edge_list(sources, destinations, None)
    G.delete_edge_list()
    with pytest.raises(libcudf.GDFError.GDFError) as excinfo:
        G.view_adj_list()
    assert excinfo.value.errcode.decode() == 'GDF_INVALID_API_CALL'

    G.add_adj_list(offsets, indices, None)
    G.delete_adj_list()
    with pytest.raises(libcudf.GDFError.GDFError) as excinfo:
        G.view_edge_list()
    assert excinfo.value.errcode.decode() == 'GDF_INVALID_API_CALL'


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_add_edge_or_adj_list_after_add_edge_or_adj_list(
        managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm_cfg.initial_pool_size = 2 << 27
    rmm.initialize()

    assert(rmm.is_initialized())

    M = utils.read_mtx_file(graph_file)
    sources = cudf.Series(M.row)
    destinations = cudf.Series(M.col)

    M = M.tocsr()
    if M is None:
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')

    offsets = cudf.Series(M.indptr)
    indices = cudf.Series(M.indices)

    G = cugraph.Graph()

    # If cugraph has at least one graph representation, adding a new graph
    # should fail to prevent a single graph object storing two different
    # graphs.

    # If cugraph has a graph edge list, adding a new graph should fail.
    G.add_edge_list(sources, destinations, None)
    with pytest.raises(libcudf.GDFError.GDFError) as excinfo:
        G.add_edge_list(sources, destinations, None)
    assert excinfo.value.errcode.decode() == 'GDF_INVALID_API_CALL'
    with pytest.raises(libcudf.GDFError.GDFError) as excinfo:
        G.add_adj_list(offsets, indices, None)
    assert excinfo.value.errcode.decode() == 'GDF_INVALID_API_CALL'
    G.delete_edge_list()

    # If cugraph has a graph adjacency list, adding a new graph should fail.
    G.add_adj_list(sources, destinations, None)
    with pytest.raises(libcudf.GDFError.GDFError) as excinfo:
        G.add_edge_list(sources, destinations, None)
    assert excinfo.value.errcode.decode() == 'GDF_INVALID_API_CALL'
    with pytest.raises(libcudf.GDFError.GDFError) as excinfo:
        G.add_adj_list(offsets, indices, None)
    assert excinfo.value.errcode.decode() == 'GDF_INVALID_API_CALL'
    G.delete_adj_list()


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_networkx_compatibility(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm_cfg.initial_pool_size = 2 << 27
    rmm.initialize()

    assert(rmm.is_initialized())

    # test from_cudf_edgelist()

    M = utils.read_mtx_file(graph_file)

    df = pd.DataFrame()
    df['source'] = pd.Series(M.row)
    df['target'] = pd.Series(M.col)
    df['weight'] = pd.Series(M.data)

    gdf = cudf.from_pandas(df)

    # cugraph.Graph() is implicitly a directed graph right at this moment, so
    # we should use nx.DiGraph() for comparison.
    Gnx = nx.from_pandas_edgelist(df,
                                  source='source',
                                  target='target',
                                  edge_attr=['weight'],
                                  create_using=nx.DiGraph)
    G = cugraph.from_cudf_edgelist(gdf,
                                   source='source',
                                   target='target',
                                   weight='weight')

    assert compare_graphs(Gnx, G)

    Gnx.clear()
    G.clear()

    # cugraph.Graph() is implicitly a directed graph right at this moment, so
    # we should use nx.DiGraph() for comparison.
    Gnx = nx.from_pandas_edgelist(df, source='source', target='target',
                                  create_using=nx.DiGraph)
    G = cugraph.from_cudf_edgelist(gdf, source='source', target='target')

    assert compare_graphs(Gnx, G)

    Gnx.clear()
    G.clear()


DATASETS2 = ['../datasets/karate',
             '../datasets/dolphins']


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS2)
def test_two_hop_neighbors(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm_cfg.initial_pool_size = 2 << 27
    rmm.initialize()

    assert(rmm.is_initialized())

    cu_M = utils.read_csv_file(graph_file+'.csv')
    sources = cu_M['0']
    destinations = cu_M['1']
    values = cu_M['2']

    G = cugraph.Graph()
    G.add_edge_list(sources, destinations, values)

    df = G.get_two_hop_neighbors()
    M = utils.read_mtx_file(graph_file+'.mtx').tocsr()
    find_two_paths(df, M)
    check_all_two_hops(df, M)


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_degree_functionality(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm_cfg.initial_pool_size = 2 << 27
    rmm.initialize()

    assert(rmm.is_initialized())

    M = utils.read_mtx_file(graph_file+'.mtx')
    cu_M = utils.read_csv_file(graph_file+'.csv')
    sources = cu_M['0']
    destinations = cu_M['1']
    values = cu_M['2']

    G = cugraph.Graph()
    G.add_edge_list(sources, destinations, values)

    Gnx = nx.DiGraph(M)

    df_in_degree = G.in_degree()
    df_out_degree = G.out_degree()
    df_degree = G.degree()

    nx_in_degree = Gnx.in_degree()
    nx_out_degree = Gnx.out_degree()
    nx_degree = Gnx.degree()

    err_in_degree = 0
    err_out_degree = 0
    err_degree = 0
    for i in range(len(df_degree)):
        if(df_in_degree['degree'][i] != nx_in_degree[i]):
            err_in_degree = err_in_degree + 1
        if(df_out_degree['degree'][i] != nx_out_degree[i]):
            err_out_degree = err_out_degree + 1
        if(df_degree['degree'][i] != nx_degree[i]):
            err_degree = err_degree + 1
    assert err_in_degree == 0
    assert err_out_degree == 0
    assert err_degree == 0


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_degrees_functionality(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm_cfg.initial_pool_size = 2 << 27
    rmm.initialize()

    assert(rmm.is_initialized())

    M = utils.read_mtx_file(graph_file+'.mtx')
    cu_M = utils.read_csv_file(graph_file+'.csv')
    sources = cu_M['0']
    destinations = cu_M['1']
    values = cu_M['2']

    G = cugraph.Graph()
    G.add_edge_list(sources, destinations, values)

    Gnx = nx.DiGraph(M)

    df = G.degrees()

    nx_in_degree = Gnx.in_degree()
    nx_out_degree = Gnx.out_degree()

    err_in_degree = 0
    err_out_degree = 0

    for i in range(len(df)):
        if(df['in_degree'][i] != nx_in_degree[i]):
            err_in_degree = err_in_degree + 1
        if(df['out_degree'][i] != nx_out_degree[i]):
            err_out_degree = err_out_degree + 1

    assert err_in_degree == 0
    assert err_out_degree == 0


'''
def test_renumber():
    source_list = ['192.168.1.1',
                   '172.217.5.238',
                   '216.228.121.209',
                   '192.16.31.23']
    dest_list = ['172.217.5.238',
                 '216.228.121.209',
                 '192.16.31.23',
                 '192.168.1.1']
    source_as_int = [
        struct.unpack('!L', socket.inet_aton(x))[0] for x in source_list
    ]
    dest_as_int = [
        struct.unpack('!L', socket.inet_aton(x))[0] for x in dest_list
    ]

    df = pd.DataFrame({
            'source_list': source_list,
            'dest_list': dest_list,
            'source_as_int': source_as_int,
            'dest_as_int': dest_as_int
            })

    G = cugraph.Graph()

    gdf = cudf.DataFrame.from_pandas(df[['source_as_int', 'dest_as_int']])

    src, dst, numbering = cugraph.renumber(gdf['source_as_int'],
                                           gdf['dest_as_int'])

    for i in range(len(source_as_int)):
        assert source_as_int[i] == numbering[src[i]]
        assert dest_as_int[i] == numbering[dst[i]]
'''


def test_renumber_negative():
    source_list = [4, 6, 8, -20, 1]
    dest_list = [1, 29, 35, 0, 77]

    df = pd.DataFrame({
        'source_list': source_list,
        'dest_list': dest_list,
    })

    gdf = cudf.DataFrame.from_pandas(df[['source_list', 'dest_list']])

    src, dst, numbering = cugraph.renumber(gdf['source_list'],
                                           gdf['dest_list'])

    for i in range(len(source_list)):
        assert source_list[i] == numbering[src[i]]
        assert dest_list[i] == numbering[dst[i]]


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_renumber_files(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm_cfg.initial_pool_size = 2 << 27
    rmm.initialize()

    assert(rmm.is_initialized())

    M = utils.read_mtx_file(graph_file)
    sources = cudf.Series(M.row)
    destinations = cudf.Series(M.col)

    translate = 1000

    source_translated = cudf.Series([x + translate for x in sources])
    dest_translated = cudf.Series([x + translate for x in destinations])

    src, dst, numbering = cugraph.renumber(source_translated, dest_translated)

    for i in range(len(sources)):
        assert sources[i] == (numbering[src[i]] - translate)
        assert destinations[i] == (numbering[dst[i]] - translate)


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_number_of_vertices(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm_cfg.initial_pool_size = 2 << 27
    rmm.initialize()

    assert(rmm.is_initialized())

    cu_M = utils.read_csv_file(graph_file+'.csv')
    sources = cu_M['0']
    destinations = cu_M['1']

    M = utils.read_mtx_file(graph_file+'.mtx')
    if M is None:
        raise TypeError('Could not read the input graph')

    # cugraph add_edge_list
    G = cugraph.Graph()
    G.add_edge_list(sources, destinations, None)
    assert(G.number_of_vertices() == M.shape[0])
