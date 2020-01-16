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

from scipy.io import mmread

import cudf
import cugraph
from cugraph.tests import utils
import rmm
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
    edgelist_df = cu_graph.view_edge_list()

    df = cudf.DataFrame()
    df['source'] = edgelist_df['src']
    df['target'] = edgelist_df['dst']
    if len(edgelist_df.columns) > 2:
        df['weight'] = edgelist_df['weights']
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

    if len(edgelist_df.columns) > 2:
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


DATASETS = ['../datasets/karate.csv',
            '../datasets/dolphins.csv',
            '../datasets/netscience.csv']


@pytest.mark.parametrize('graph_file', DATASETS)
def test_read_csv_for_nx(graph_file):

    Mnew = utils.read_csv_for_nx(graph_file, read_weights_in_sp=False)
    if Mnew is None:
        raise TypeError('Could not read the input graph')
    if Mnew.shape[0] != Mnew.shape[1]:
        raise TypeError('Shape is not square')

    Mold = mmread(graph_file.replace('.csv', '.mtx')).asfptype()

    minnew = Mnew.data.min()
    minold = Mold.data.min()
    epsilon = min(minnew, minold) / 1000.0

    mdiff = abs(Mold - Mnew)
    mdiff.data[mdiff.data < epsilon] = 0
    mdiff.eliminate_zeros()

    assert Mold.nnz == Mnew.nnz
    assert Mold.shape == Mnew.shape
    assert mdiff.nnz == 0


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_add_edge_list_to_adj_list(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    cu_M = utils.read_csv_file(graph_file)

    M = utils.read_csv_for_nx(graph_file).tocsr()
    if M is None:
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')

    offsets_exp = M.indptr
    indices_exp = M.indices

    # cugraph add_egde_list to_adj_list call
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1')
    offsets_cu, indices_cu, values_cu = G.view_adj_list()
    assert compare_offsets(offsets_cu, offsets_exp)
    assert compare_series(indices_cu, indices_exp)
    assert values_cu is None
