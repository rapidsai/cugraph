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


def networkx_weak_call(M):
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


def cugraph_weak_call(cu_M):
    # cugraph Pagerank Call
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source='0', target='1')
    t1 = time.time()
    df = cugraph.weakly_connected_components(G)
    t2 = time.time() - t1
    print('Time : '+str(t2))

    result = df['labels'].to_array()

    labels = sorted(result)
    return labels


def networkx_strong_call(M):
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
    result = nx.strongly_connected_components(Gnx)
    t2 = time.time() - t1

    print('Time : ' + str(t2))

    labels = sorted(result)
    return labels


def cugraph_strong_call(cu_M):
    # cugraph Pagerank Call
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source='0', target='1')
    t1 = time.time()
    df = cugraph.strongly_connected_components(G)
    t2 = time.time() - t1
    print('Time : '+str(t2))

    result = df['labels'].to_array()

    labels = sorted(result)
    return labels


# these should come w/ cugraph/python:
#
DATASETS = ['../datasets/dolphins.csv',
            '../datasets/netscience.csv']

STRONGDATASETS = ['../datasets/dolphins.csv',
                  '../datasets/netscience.csv',
                  '../datasets/email-Eu-core.csv']


# vcount how many `val`s in ls container:
#
def counter_f(ls, val):
    return sum(1 for x in ls if x == val)


# return number of uniques values in lst container:
#
def get_n_uniqs(lst):
    return len(set(lst))


# gets unique values of list and then counts the
# occurences of each unique value within list;
# note: because of using set(), the "keys"
# (unique values) will be sorted in set(lst)
#
def get_uniq_counts(lst):
    return [counter_f(lst, uniq_val) for uniq_val in set(lst)]


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_weak_cc(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    M = utils.read_csv_for_nx(graph_file)
    netx_labels = networkx_weak_call(M)

    cu_M = utils.read_csv_file(graph_file)
    cugraph_labels = cugraph_weak_call(cu_M)

    # NetX returns a list of components, each component being a
    # collection (set{}) of vertex indices;
    #
    # while cugraph returns a component label for each vertex;

    nx_n_components = len(netx_labels)
    cg_n_components = get_n_uniqs(cugraph_labels)

    assert nx_n_components == cg_n_components

    lst_nx_components_lens = [len(c) for c in sorted(netx_labels, key=len)]

    # get counts of uniques:
    #
    lst_cg_components_lens = sorted(get_uniq_counts(cugraph_labels))

    assert lst_nx_components_lens == lst_cg_components_lens


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', STRONGDATASETS)
def test_strong_cc(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    M = utils.read_csv_for_nx(graph_file)
    netx_labels = networkx_strong_call(M)

    cu_M = utils.read_csv_file(graph_file)
    cugraph_labels = cugraph_strong_call(cu_M)

    # NetX returns a list of components, each component being a
    # collection (set{}) of vertex indices;
    #
    # while cugraph returns a component label for each vertex;

    nx_n_components = len(netx_labels)
    cg_n_components = get_n_uniqs(cugraph_labels)

    assert nx_n_components == cg_n_components

    lst_nx_components_lens = [len(c) for c in sorted(netx_labels, key=len)]

    # get counts of uniques:
    #
    lst_cg_components_lens = sorted(get_uniq_counts(cugraph_labels))

    assert lst_nx_components_lens == lst_cg_components_lens
