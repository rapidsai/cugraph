# Copyright (c) 2019-2020, NVIDIA CORPORATION.:
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

import pytest

import cugraph
from cugraph.tests import utils
import random
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

# NOTE: Endpoint parameter is not currently being tested, there could be a test
#       to verify that python raise an error if it is used
# =============================================================================
# Parameters
# =============================================================================
DIRECTED_GRAPH_OPTIONS = [False, True]
DEFAULT_EPSILON = 0.0001
IMPLEMENTATION_OPTIONS = ['default', 'gunrock']

# TINY_DATASETS = ['../datasets/karate.csv']

# UNRENUMBERED_DATASETS = ['../datasets/karate.csv']

# SMALL_DATASETS = ['../datasets/netscience.csv']

SUBSET_SIZE_OPTIONS = [4]
SUBSET_SEED_OPTIONS = [42]

# NOTE: The following is not really being exploited in the tests as the
# datasets that are used are too small to compare, but it ensures that both
# path are actually sane
RESULT_DTYPE_OPTIONS = [np.float32, np.float64]


# =============================================================================
# Comparison functions
# =============================================================================
def build_graphs(graph_file, directed=True):
    # cugraph
    cu_M = utils.read_csv_file(graph_file)
    G = cugraph.DiGraph() if directed else cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1')
    G.view_adj_list()  # Enforce generation before computation

    # networkx
    M = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(M, create_using=(nx.DiGraph() if directed
                                                   else nx.Graph()),
                                  source='0', target='1')
    return G, Gnx


def calc_betweenness_centrality(graph_file, directed=True, normalized=False,
                                weight=None, endpoints=False,
                                k=None, seed=None, implementation=None,
                                result_dtype=np.float32):
    """ Generate both cugraph and networkx betweenness centrality

    Parameters
    ----------
    graph_file : string
        Path to COO Graph representation in .csv format

    directed : bool, optional, default=True

    normalized : bool
        True: Normalize Betweenness Centrality scores
        False: Scores are left unnormalized

    k : int or None, optional, default=None
        int:  Number of sources  to sample  from
        None: All sources are used to compute

    seed : int or None, optional, default=None
        Seed for random sampling  of the starting point

    implementation : string or None, optional, default=None
        There are 2 possibilities 'default' and 'gunrock', if None falls back
        into 'default'

    Returns
    -------
        cu_bc : dict
            Each key is the vertex identifier, each value is the betweenness
            centrality score obtained from cugraph betweenness_centrality
        nx_bc : dict
            Each key is the vertex identifier, each value is the betweenness
            centrality score obtained from networkx betweenness_centrality
    """
    G, Gnx = build_graphs(graph_file, directed=directed)
    calc_func = None
    if k is not None and seed is not None:
        calc_func = _calc_bc_subset
    elif k is not None:
        calc_func = _calc_bc_subset_fixed
    else:  # We processed to a comparison using every sources
        calc_func = _calc_bc_full
    cu_bc, nx_bc = calc_func(G, Gnx, normalized=normalized, weight=weight,
                             endpoints=endpoints, k=k, seed=seed,
                             implementation=implementation,
                             result_dtype=result_dtype)

    return cu_bc, nx_bc


def _calc_bc_subset(G, Gnx, normalized, weight, endpoints, k, seed,
                    implementation, result_dtype):
    # NOTE: Networkx API does not allow passing a list of vertices
    # And the sampling is operated on Gnx.nodes() directly
    # We first mimic acquisition of the nodes to compare with same sources
    random.seed(seed)  # It will be called again in nx's call
    sources = random.sample(Gnx.nodes(), k)
    df = cugraph.betweenness_centrality(G, normalized=normalized,
                                        weight=weight,
                                        endpoints=endpoints,
                                        k=sources,
                                        implementation=implementation,
                                        result_dtype=result_dtype)
    nx_bc = nx.betweenness_centrality(Gnx, normalized=normalized, k=k,
                                      seed=seed)
    cu_bc = {key: score for key, score in
             zip(df['vertex'].to_array(),
                 df['betweenness_centrality'].to_array())}
    return cu_bc, nx_bc


def _calc_bc_subset_fixed(G, Gnx, normalized, weight, endpoints, k, seed,
                          implementation, result_dtype):
    assert isinstance(k, int), "This test is meant for verifying coherence " \
                               "when k is given as an int"
    # In the fixed set we compare cu_bc against itself as we random.seed(seed)
    # on the same seed and then sample on the number of vertices themselves
    if seed is None:
        seed = 123  # random.seed(None) uses time, but we want same sources
    random.seed(seed)  # It will be called again in cugraph's call
    sources = random.sample(range(G.number_of_vertices()), k)
    # The first call is going to proceed to the random sampling in the same
    # fashion as the lines above
    df = cugraph.betweenness_centrality(G, k=k, normalized=normalized,
                                        weight=weight,
                                        endpoints=endpoints,
                                        implementation=implementation,
                                        seed=seed,
                                        result_dtype=result_dtype)
    # The second call is going to process source that were already sampled
    # We set seed to None as k : int, seed : not none should not be normal
    # behavior
    df2 = cugraph.betweenness_centrality(G, k=sources, normalized=normalized,
                                         weight=weight,
                                         endpoints=endpoints,
                                         implementation=implementation,
                                         seed=None,
                                         result_dtype=result_dtype)
    cu_bc = {key: score for key, score in
             zip(df['vertex'].to_array(),
                 df['betweenness_centrality'].to_array())}
    cu_bc2 = {key: score for key, score in
              zip(df2['vertex'].to_array(),
                  df2['betweenness_centrality'].to_array())}

    return cu_bc, cu_bc2


def _calc_bc_full(G, Gnx, normalized, weight, endpoints, implementation,
                  k, seed,
                  result_dtype):
    df = cugraph.betweenness_centrality(G, normalized=normalized,
                                        weight=weight,
                                        endpoints=endpoints,
                                        implementation=implementation,
                                        result_dtype=result_dtype)
    assert df['betweenness_centrality'].dtype == result_dtype,  \
        "'betweenness_centrality' column has not the expected type"
    nx_bc = nx.betweenness_centrality(Gnx, normalized=normalized,
                                      weight=weight,
                                      endpoints=endpoints)

    cu_bc = {key: score for key, score in
             zip(df['vertex'].to_array(),
                 df['betweenness_centrality'].to_array())}
    return cu_bc, nx_bc


# =============================================================================
# Utils
# =============================================================================
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
        True: Result and expected are close to each other
        False: Otherwise
    """
    close = np.isclose(result, expected, rtol=epsilon)
    return close


# NOTE: We assume that both cugraph and networkx are generating dicts with
#       all the sources, thus we can compare all of them
def compare_scores(cu_bc, ref_bc, epsilon=DEFAULT_EPSILON):
    missing_key_error = 0
    score_mismatch_error = 0
    for vertex in ref_bc:
        if vertex in cu_bc:
            result = cu_bc[vertex]
            expected = ref_bc[vertex]
            if not compare_single_score(result, expected, epsilon=epsilon):
                score_mismatch_error += 1
                print("ERROR: vid = {}, cu = {}, "
                      "nx = {}".format(vertex, result, expected))
        else:
            missing_key_error += 1
            print("[ERROR] Missing vertex {vertex}".format(vertex=vertex))
    assert missing_key_error == 0, "Some vertices were missing"
    assert score_mismatch_error == 0, "Some scores were not close enough"


def prepare_test():
    gc.collect()


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.parametrize('graph_file', utils.TINY_DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('implementation', IMPLEMENTATION_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
def test_betweenness_centrality_normalized_tiny(graph_file,
                                                directed, implementation,
                                                result_dtype):
    """Test Normalized Betweenness Centrality"""
    prepare_test()
    cu_bc, nx_bc = calc_betweenness_centrality(graph_file, directed=directed,
                                               normalized=True,
                                               implementation=implementation,
                                               result_dtype=result_dtype)
    compare_scores(cu_bc, nx_bc)


@pytest.mark.parametrize('graph_file', utils.TINY_DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('implementation', IMPLEMENTATION_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
def test_betweenness_centrality_unnormalized_tiny(graph_file,
                                                  directed, implementation,
                                                  result_dtype):
    """Test Unnormalized Betweenness Centrality"""
    prepare_test()
    cu_bc, nx_bc = calc_betweenness_centrality(graph_file, directed=directed,
                                               normalized=False,
                                               implementation=implementation,
                                               result_dtype=result_dtype)
    compare_scores(cu_bc, nx_bc)


@pytest.mark.parametrize('graph_file', utils.SMALL_DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('implementation', IMPLEMENTATION_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
def test_betweenness_centrality_normalized_small(graph_file,
                                                 directed, implementation,
                                                 result_dtype):
    """Test Unnormalized Betweenness Centrality"""
    prepare_test()
    cu_bc, nx_bc = calc_betweenness_centrality(graph_file, directed=directed,
                                               normalized=True,
                                               implementation=implementation,
                                               result_dtype=result_dtype)
    compare_scores(cu_bc, nx_bc)


@pytest.mark.parametrize('graph_file', utils.SMALL_DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('implementation', IMPLEMENTATION_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
def test_betweenness_centrality_unnormalized_small(graph_file,
                                                   directed, implementation,
                                                   result_dtype):
    """Test Unnormalized Betweenness Centrality"""
    prepare_test()
    cu_bc, nx_bc = calc_betweenness_centrality(graph_file, directed=directed,
                                               normalized=False,
                                               implementation=implementation,
                                               result_dtype=result_dtype)
    compare_scores(cu_bc, nx_bc)


@pytest.mark.parametrize('graph_file', utils.SMALL_DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('subset_size', SUBSET_SIZE_OPTIONS)
@pytest.mark.parametrize('subset_seed', SUBSET_SEED_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
def test_betweenness_centrality_normalized_subset_small(graph_file,
                                                        directed,
                                                        subset_size,
                                                        subset_seed,
                                                        result_dtype):
    """Test Unnormalized Betweenness Centrality using a subset

    Only k sources are considered for an approximate Betweenness Centrality
    """
    prepare_test()
    cu_bc, nx_bc = calc_betweenness_centrality(graph_file,
                                               directed=directed,
                                               normalized=True,
                                               k=subset_size,
                                               seed=subset_seed,
                                               result_dtype=result_dtype)
    compare_scores(cu_bc, nx_bc)


# NOTE: This test should only be execute on unrenumbered datasets
#       the function operating the comparison inside is first proceeding
#       to a random sampling over the number of vertices (thus direct offsets)
#       in the graph structure instead of actual vertices identifiers
@pytest.mark.parametrize('graph_file', utils.UNRENUMBERED_DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('subset_size', SUBSET_SIZE_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
def test_betweenness_centrality_normalized_fixed_sample(graph_file,
                                                        directed,
                                                        subset_size,
                                                        result_dtype):
    """Test Unnormalized Betweenness Centrality using a subset

    Only k sources are considered for an approximate Betweenness Centrality
    """
    prepare_test()
    cu_bc, nx_bc = calc_betweenness_centrality(graph_file,
                                               directed=directed,
                                               normalized=True,
                                               k=subset_size,
                                               seed=None,
                                               result_dtype=result_dtype)
    compare_scores(cu_bc, nx_bc)


@pytest.mark.parametrize('graph_file', utils.SMALL_DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('subset_size', SUBSET_SIZE_OPTIONS)
@pytest.mark.parametrize('subset_seed', SUBSET_SEED_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
def test_betweenness_centrality_unnormalized_subset_small(graph_file,
                                                          directed,
                                                          subset_size,
                                                          subset_seed,
                                                          result_dtype):
    """Test Unnormalized Betweenness Centrality on Graph on subset

    Only k sources are considered for an approximate Betweenness Centrality
    """
    prepare_test()
    cu_bc, nx_bc = calc_betweenness_centrality(graph_file,
                                               directed=directed,
                                               normalized=False,
                                               k=subset_size,
                                               seed=subset_seed,
                                               result_dtype=result_dtype)
    compare_scores(cu_bc, nx_bc)


@pytest.mark.parametrize('graph_file', utils.TINY_DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
def test_betweenness_centrality_invalid_implementation(graph_file,
                                                       directed,
                                                       result_dtype):
    """Test calls betwenness_centrality with an invalid implementation name"""
    prepare_test()
    with pytest.raises(ValueError):
        cu_bc, nx_bc = calc_betweenness_centrality(graph_file,
                                                   directed=directed,
                                                   implementation="invalid",
                                                   result_dtype=result_dtype)


@pytest.mark.parametrize('graph_file', utils.TINY_DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
def test_betweenness_centrality_gunrock_subset(graph_file,
                                               directed,
                                               result_dtype):
    """Test calls betwenness_centrality with subset and gunrock"""
    prepare_test()
    with pytest.raises(ValueError):
        cu_bc, nx_bc = calc_betweenness_centrality(graph_file,
                                                   directed=directed,
                                                   normalized=False,
                                                   k=1,
                                                   implementation="gunrock",
                                                   result_dtype=result_dtype)


@pytest.mark.parametrize('graph_file', utils.TINY_DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
def test_betweenness_centrality_unnormalized_endpoints_except(graph_file,
                                                              directed,
                                                              result_dtype):
    """Test calls betwenness_centrality unnormalized + endpoints"""
    prepare_test()
    with pytest.raises(NotImplementedError):
        cu_bc, nx_bc = calc_betweenness_centrality(graph_file,
                                                   normalized=False,
                                                   endpoints=True,
                                                   directed=directed,
                                                   result_dtype=result_dtype)


@pytest.mark.parametrize('graph_file', utils.TINY_DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
def test_betweenness_centrality_normalized_endpoints_except(graph_file,
                                                            directed,
                                                            result_dtype):
    """Test calls betwenness_centrality normalized + endpoints"""
    prepare_test()
    with pytest.raises(NotImplementedError):
        cu_bc, nx_bc = calc_betweenness_centrality(graph_file,
                                                   normalized=True,
                                                   endpoints=True,
                                                   directed=directed,
                                                   result_dtype=result_dtype)


@pytest.mark.parametrize('graph_file', utils.TINY_DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
def test_betweenness_centrality_unnormalized_weight_except(graph_file,
                                                           directed,
                                                           result_dtype):
    """Test calls betwenness_centrality unnormalized + weight"""
    prepare_test()
    with pytest.raises(NotImplementedError):
        cu_bc, nx_bc = calc_betweenness_centrality(graph_file,
                                                   normalized=False,
                                                   weight=True,
                                                   directed=directed,
                                                   result_dtype=result_dtype)


@pytest.mark.parametrize('graph_file', utils.TINY_DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
def test_betweenness_centrality_normalized_weight_except(graph_file,
                                                         directed,
                                                         result_dtype):
    """Test calls betwenness_centrality normalized + weight"""
    prepare_test()
    with pytest.raises(NotImplementedError):
        cu_bc, nx_bc = calc_betweenness_centrality(graph_file,
                                                   normalized=True,
                                                   weight=True,
                                                   directed=directed,
                                                   result_dtype=result_dtype)


@pytest.mark.parametrize('graph_file', utils.TINY_DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
def test_betweenness_centrality_invalid_dtype(graph_file, directed):
    """Test calls betwenness_centrality normalized + weight"""
    prepare_test()
    with pytest.raises(TypeError):
        cu_bc, nx_bc = calc_betweenness_centrality(graph_file,
                                                   normalized=True,
                                                   result_dtype=str,
                                                   directed=directed)
