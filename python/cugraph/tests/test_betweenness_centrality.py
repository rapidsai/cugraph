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
import random
import time # To add call timer

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

def compare_close_scores(scores, idx, epsilon):
    """
    Compare value in score at given index with relative error

    Parameters
    ----------
    scores : DataFrame
        contains 'cu' and 'nx' columns which are the values to compare
    idx : int
        row index of the DataFrame
    epsilon : floating point
        indicates relative error tolerated

    Returns
    -------
    err : int
        1: If there is a mismatch
        0: Otherwise
    """
    err = 0
    if (scores['cu'][idx] < (scores['nx'][idx] * (1 - epsilon)) or
        scores['cu'][idx] > (scores['nx'][idx] * (1 + epsilon))):
        err = err + 1
        print('ERROR: vid = {}, cu = {}, nx = {}'.format(scores['vertex'][idx],
                                                         scores['cu'][idx],
                                                         scores['nx'][idx]))
    #print("Abs diff:", abs(scores["cu"][idx] - scores["nx"][idx]))
    return err

def calc_betweenness_centrality(graph_file, normalized=True):
    cu_M = utils.read_csv_file(graph_file)
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1')

    df = cugraph.betweenness_centrality(G, normalized=normalized)

    NM = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(NM, create_using=nx.DiGraph(),
                                  source='0', target='1')

    nb = nx.betweenness_centrality(Gnx, normalized=normalized)

    pdf = [nb[k] for k in sorted(nb.keys())]
    df['nx'] = pdf
    df = df.rename({'betweenness_centrality': 'cu'})
    return df

# TODO(xcadet)  Fix the following part with the number of sources
# TODO(xcadet) Clean this part
def calc_betweenness_centrality_k(graph_file, normalized=True):
    # For this case we need to swap Gnx and G generation,
    # In order to ensure comparability of the resultS with a subsample
    NM = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(NM, create_using=nx.DiGraph(),
                                  source='0', target='1')
    number_of_sources = int(len(Gnx) * 0.05)
    number_of_sources = 4 # For GAP equivalence
    seed = 42
    random.seed(seed)
    vertices = random.sample(Gnx.nodes(), number_of_sources)
    print("[DBG]Processing vertices:", vertices)
    print("[DBG]Normalized:", normalized)
    random.seed(seed)
    second_vertices = random.sample(Gnx.nodes(), number_of_sources)
    print("[DBG]Processing second vertices:", second_vertices)
    start = time.perf_counter()
    nb = nx.betweenness_centrality(Gnx, normalized=normalized, k=number_of_sources, seed=seed)
    end = time.perf_counter()
    print("[DBG]nx: {}".format(end - start))

    cu_M = utils.read_csv_file(graph_file)
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1')
    G.view_adj_list() # Enforce Adjacency

    print("[DBG] Is Renumbered ?", G.renumbered)
    start = time.perf_counter()
    df = cugraph.betweenness_centrality(G, normalized=normalized, k=vertices)
    end = time.perf_counter()
    print("[DBG]cu: {}".format(end - start))

    pdf = [nb[k] for k in sorted(nb.keys())]
    df['nx'] = pdf
    df = df.rename({'betweenness_centrality': 'cu'})
    return df

TINY_DATASETS = ['../datasets/karate.csv',
                 '../datasets/dolphins.csv',
                '../datasets/polbooks.csv']
SMALL_DATASETS = ['../datasets/netscience.csv']


@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', TINY_DATASETS)
def test_betweenness_centrality(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool
    )

    assert(rmm.is_initialized())

    scores = calc_betweenness_centrality(graph_file)

    err = 0
    epsilon = 0.0001
    for idx in range(len(scores)):
        err += compare_close_scores(scores, idx, epsilon)
    assert err == 0

@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', TINY_DATASETS)
def test_betweenness_centrality_unnormalized(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool
    )

    assert(rmm.is_initialized())

    scores = calc_betweenness_centrality(graph_file, False)

    err = 0
    epsilon = 0.0001


    for idx in range(len(scores)):
        err += compare_close_scores(scores, idx, epsilon)
    assert err == 0

@pytest.mark.small
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', SMALL_DATASETS)
def test_betweenness_centrality_unnormalized_5percent(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool
    )

    assert(rmm.is_initialized())

    scores = calc_betweenness_centrality_k(graph_file, False)

    err = 0
    epsilon = 0.0001

    for idx in range(len(scores)):
        err += compare_close_scores(scores, idx, epsilon)
    assert err == 0

#LARGE_DATASETS = ['/datasets/GAP/GAP-road.csv']
LARGE_DATASETS = ['../datasets/road_central.csv']
@pytest.mark.large
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', LARGE_DATASETS)
def test_betweenness_centrality_unnormalized_5percent(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool
    )

    assert(rmm.is_initialized())

    scores = calc_betweenness_centrality_k(graph_file, False)

    err = 0
    epsilon = 0.0001

    for idx in range(len(scores)):
        err += compare_close_scores(scores, idx, epsilon)
    assert err == 0