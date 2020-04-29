# Copyright (c) 2019, NVIDIA CORPORATION.:
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
import time
import numpy as np

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
#===============================================================================
# Parameters
#===============================================================================
RMM_MANAGED_MEMORY_OPTIONS  = [False, True]
RMM_POOL_ALLOCATOR_OPTIONS  = [False, True]
DIRECTED_GRAPH_OPTIONS      = [False, True]
DEFAULT_EPSILON             = 0.0001

TINY_DATASETS               = ['../datasets/karate.csv',
                              '../datasets/dolphins.csv',
                              '../datasets/polbooks.csv']

SMALL_DATASETS              = ['../datasets/netscience.csv']

LARGE_DATASETS              = ['../datasets/road_central.csv']

SUBSET_SIZE_OPTIONS         = [1]
SUBSET_SEED_OPTIONS         = [42]

#===============================================================================
# Comparison functions
#===============================================================================
def build_graphs(graph_file, directed=True):
    # cugraph
    cu_M = utils.read_csv_file(graph_file)
    G = cugraph.DiGraph() if directed else cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1')
    G.view_adj_list() # Enforce generation before computation

    # networkx
    M = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(M, create_using=(nx.DiGraph() if directed
                                                   else nx.Graph()),
                                  source='0', target='1')
    return G, Gnx

def calc_betweenness_centrality(graph_file, directed=True, normalized=False,
                                k=None, seed=None):
    """ Generate both cugraph and networkx betweenness centrality

    Parameters
    ----------
    graph_file : string
        Path to COO Graph representation in .csv format

    directed : bool, optional, default=True

    normalized : bool
        True: Normalize Betweenness Centrality scores
        False: Scores are left unormalized

    Returns
    -------
        df : cudf.DataFrame
            Contains 'vertex', 'cu' and 'nx' columns
            'vertex': Indices of the vertices
            'cu': Betweenness Centrality scores obtained with cugraph
            'nx': Betweenness Centrality scores obtained with networkx
    """
    G, Gnx = build_graphs(graph_file, directed=directed)
    print("[DBG] Directed:", directed, "cu:", type(G), "nx:", type(Gnx))
    print("[DBG] Normalized:", normalized)

    if k is not None and seed is not None:
        df, nb = _calc_betweenness_centrality_subset(G, Gnx,
                                                     normalized=normalized, k=k,
                                                     seed=seed)
    else:
        df, nb = _calc_betweenness_centrality_full(G, Gnx, normalized=normalized)

    pdf = [nb[k] for k in sorted(nb.keys())]
    df['nx'] = pdf
    df = df.rename({'betweenness_centrality': 'cu'})
    return df

def _calc_betweenness_centrality_subset(G, Gnx, normalized, k, seed):
    # NOTE: Networkx API does not allow passing a list of vertices
    # And the sampling is operated on Gnx.nodes() directly
    # We first mimic acquisition of the nodes to compare with same sources
    random.seed(seed) # It will be called again on nx call
    sources = random.sample(Gnx.nodes(), k)
    df = cugraph.betweenness_centrality(G, normalized=normalized, k=sources)
    nb = nx.betweenness_centrality(Gnx, normalized=normalized, k=k, seed=seed)
    return df, nb

def _calc_betweenness_centrality_full(G, Gnx, normalized):
    df = cugraph.betweenness_centrality(G, normalized=normalized)
    nb = nx.betweenness_centrality(Gnx, normalized=normalized)
    return df, nb

#===============================================================================
# Utils
#===============================================================================
def prepare_rmm(managed_memory, pool_allocator):
    gc.collect()
    rmm.reinitialize(
        managed_memory=managed_memory,
        pool_allocator=pool_allocator,
    )
    assert(rmm.is_initialized)

def compare_single_score(result, expected, epsilon):
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
    close : bool
        True: Result and expected are close to each oter
        False: Ohterwise
    """
    close = np.isclose(result, expected, rtol=epsilon)
    return close

def compare_scores(scores, epsilon=DEFAULT_EPSILON):
    err = 0
    for idx in range(len(scores)):
        score_cu =  scores['cu'][idx]
        score_nx = scores['nx'][idx]
        if not compare_single_score(score_cu, score_nx, epsilon=epsilon):
            err += 1
            print('ERROR: id = {}, vid = {}, cu = {}, nx = {}'.format(idx,
                                                            scores['vertex'][idx],
                                                            score_cu,
                                                            score_nx))
    assert err == 0, "Some scores were not close enough"

#===============================================================================
# Tests
#===============================================================================
#@pytest.mark.parametrize('managed, pool',
                         #list(product(RMM_MANAGED_MEMORY_OPTIONS,
                                      #RMM_POOL_ALLOCATOR_OPTIONS)))
#@pytest.mark.parametrize('graph_file', TINY_DATASETS)
#@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
#def test_betweenness_centrality(managed, pool, graph_file, directed):
    #"""Test Normalized Betweenness Centrality"""
    #prepare_rmm(managed, pool)
    #scores = calc_betweenness_centrality(graph_file, directed=directed,
                                         #normalized=True)
    #compare_scores(scores)

#@pytest.mark.parametrize('managed, pool',
                         #list(product(RMM_MANAGED_MEMORY_OPTIONS,
                                      #RMM_POOL_ALLOCATOR_OPTIONS)))
#@pytest.mark.parametrize('graph_file', TINY_DATASETS)
#@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
#def test_betweenness_centrality_unnormalized(managed, pool, graph_file, directed):
    #"""Test Unnormalized Betweenness Centrality"""
    #prepare_rmm(managed, pool)
    #scores = calc_betweenness_centrality(graph_file, directed=directed,
                                        #normalized=False)
    #compare_scores(scores)

#@pytest.mark.parametrize('managed, pool',
                         #list(product(RMM_MANAGED_MEMORY_OPTIONS,
                                      #RMM_POOL_ALLOCATOR_OPTIONS)))
#@pytest.mark.parametrize('graph_file', SMALL_DATASETS)
#@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
#def test_betweenness_centrality_unnormalized(managed, pool, graph_file, directed):
    #"""Test Unnormalized Betweenness Centrality"""
    #prepare_rmm(managed, pool)
    #scores = calc_betweenness_centrality(graph_file, directed=directed,
                                        #normalized=False)
    #compare_scores(scores)



#@pytest.mark.parametrize('managed, pool',
                         #list(product(RMM_MANAGED_MEMORY_OPTIONS,
                                      #RMM_POOL_ALLOCATOR_OPTIONS)))
#@pytest.mark.parametrize('graph_file', SMALL_DATASETS)
#@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
#@pytest.mark.parametrize('subset_size', SUBSET_SIZE_OPTIONS)
#@pytest.mark.parametrize('subset_seed', SUBSET_SEED_OPTIONS)
#def test_betweenness_centrality_unnormalized_subset(managed, pool,
                                                    #graph_file,
                                                    #directed,
                                                    #subset_size, subset_seed):
    #"""Test Unnormalized Betweenness Centrality on Directed Graph on subset

    #Only k sources are considered for an approximate Betweenness Centrality
    #"""
    #prepare_rmm(managed, pool)
    #scores = calc_betweenness_centrality(graph_file,
                                         #directed=directed,
                                         #normalized=False,
                                         #k=subset_size,
                                         #seed=subset_seed)
    #compare_scores(scores)

@pytest.mark.parametrize('managed, pool',
                         list(product(RMM_MANAGED_MEMORY_OPTIONS,
                                      RMM_POOL_ALLOCATOR_OPTIONS)))
#@pytest.mark.parametrize('graph_file', ["../datasets/road_central.csv"])
@pytest.mark.parametrize('graph_file', ["../datasets/cti.csv"])
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('subset_size', SUBSET_SIZE_OPTIONS)
@pytest.mark.parametrize('subset_seed', SUBSET_SEED_OPTIONS)
def test_betweenness_centrality_unnormalized_subset(managed, pool,
                                                    graph_file,
                                                    directed,
                                                    subset_size, subset_seed):
    """Test Unnormalized Betweenness Centrality on Directed Graph on subset

    Only k sources are considered for an approximate Betweenness Centrality
    """
    prepare_rmm(managed, pool)
    scores = calc_betweenness_centrality(graph_file,
                                         directed=directed,
                                         normalized=False,
                                         k=subset_size,
                                         seed=subset_seed)
    compare_scores(scores)