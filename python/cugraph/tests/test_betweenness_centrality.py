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

import dask_cuda
import dask.distributed

import cugraph
from cugraph.tests import utils
import random
import numpy as np
import cupy
import cudf
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

# NOTE: Endpoint parameter is not currently being tested, there could be a test
#       to verify that python raise an error if it is used
# =============================================================================
# Parameters
# =============================================================================
DIRECTED_GRAPH_OPTIONS = [False, True]
ENDPOINTS_OPTIONS = [False, True]
NORMALIZED_OPTIONS = [False, True]
DEFAULT_EPSILON = 0.0001

OPG_OPTIONS = [None, 1, 2]

DATASETS = ['../datasets/karate.csv',
            '../datasets/netscience.csv']

UNRENUMBERED_DATASETS = ['../datasets/karate.csv']

SUBSET_SIZE_OPTIONS = [4, None]
SUBSET_SEED_OPTIONS = [42]

# NOTE: The following is not really being exploited in the tests as the
# datasets that are used are too small to compare, but it ensures that both
# path are actually sane
RESULT_DTYPE_OPTIONS = [np.float32, np.float64]


# =============================================================================
# Comparison functions
# =============================================================================
def calc_betweenness_centrality(graph_file, directed=True,
                                k=None, normalized=False,
                                weight=None, endpoints=False,
                                seed=None,
                                result_dtype=np.float64,
                                use_k_full=False):
    """ Generate both cugraph and networkx betweenness centrality

    Parameters
    ----------
    graph_file : string
        Path to COO Graph representation in .csv format

    directed : bool, optional, default=True

    k : int or None, optional, default=None
        int:  Number of sources  to sample  from
        None: All sources are used to compute

    normalized : bool
        True: Normalize Betweenness Centrality scores
        False: Scores are left unnormalized

    weight : cudf.DataFrame:
        Not supported as of 06/2020

    endpoints : bool
        True: Endpoints are included when computing scores
        False: Endpoints are not considered

    seed : int or None, optional, default=None
        Seed for random sampling  of the starting point

    result_dtype :  numpy.dtype
        Expected type of the result, either np.float32 or np.float64

    use_k_full : bool
        When True, if k is None replaces k by the number of sources of the
        Graph

    Returns
    -------

    sorted_df : cudf.DataFrame
        Contains 'vertex' and  'cu_bc' 'ref_bc' columns,  where 'cu_bc'
        and 'ref_bc' are the two betweenness centrality scores to compare.
        The dataframe is expected to be sorted based on 'vertex', so that we
        can use cupy.isclose to compare the scores.
    """
    G, Gnx = utils.build_cu_and_nx_graphs(graph_file, directed=directed)
    calc_func = None
    if k is not None and seed is not None:
        calc_func = _calc_bc_subset
    elif k is not None:
        calc_func = _calc_bc_subset_fixed
    else:  # We processed to a comparison using every sources
        if use_k_full:
            k = Gnx.number_of_nodes()
        calc_func = _calc_bc_full
    sorted_df = calc_func(G,
                          Gnx,
                          k=k,
                          normalized=normalized,
                          weight=weight,
                          endpoints=endpoints,
                          seed=seed,
                          result_dtype=result_dtype)

    return sorted_df


def _calc_bc_subset(G, Gnx, normalized, weight, endpoints, k, seed,
                    result_dtype):
    # NOTE: Networkx API does not allow passing a list of vertices
    # And the sampling is operated on Gnx.nodes() directly
    # We first mimic acquisition of the nodes to compare with same sources
    random.seed(seed)  # It will be called again in nx's call
    sources = random.sample(Gnx.nodes(), k)
    df = cugraph.betweenness_centrality(G,
                                        k=sources,
                                        normalized=normalized,
                                        weight=weight,
                                        endpoints=endpoints,
                                        result_dtype=result_dtype)
    sorted_df = df.sort_values("vertex").rename({"betweenness_centrality":
                                                 "cu_bc"})

    nx_bc = nx.betweenness_centrality(Gnx,
                                      k=k,
                                      normalized=normalized,
                                      weight=weight,
                                      endpoints=endpoints,
                                      seed=seed)
    _, nx_bc = zip(*sorted(nx_bc.items()))
    nx_df = cudf.DataFrame({"ref_bc": nx_bc})

    merged_sorted_df = cudf.concat([sorted_df, nx_df], axis=1, sort=False)

    return merged_sorted_df


def _calc_bc_subset_fixed(G, Gnx, normalized, weight, endpoints, k, seed,
                          result_dtype):
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
    df = cugraph.betweenness_centrality(G,
                                        k=k,
                                        normalized=normalized,
                                        weight=weight,
                                        endpoints=endpoints,
                                        seed=seed,
                                        result_dtype=result_dtype)
    sorted_df = df.sort_values("vertex").rename({"betweenness_centrality":
                                                 "cu_bc"})

    # The second call is going to process source that were already sampled
    # We set seed to None as k : int, seed : not none should not be normal
    # behavior
    df2 = cugraph.betweenness_centrality(G,
                                         k=sources,
                                         normalized=normalized,
                                         weight=weight,
                                         endpoints=endpoints,
                                         seed=None,
                                         result_dtype=result_dtype)
    sorted_df2 = df2.sort_values("vertex").rename({"betweenness_centrality":
                                                   "ref_bc"})

    merged_sorted_df = cudf.concat([sorted_df, sorted_df2["ref_bc"]], axis=1,
                                   sort=False)

    return merged_sorted_df


def _calc_bc_full(G, Gnx, normalized, weight, endpoints,
                  k, seed,
                  result_dtype):
    df = cugraph.betweenness_centrality(G,
                                        k=k,
                                        normalized=normalized,
                                        weight=weight,
                                        endpoints=endpoints,
                                        result_dtype=result_dtype)
    assert df['betweenness_centrality'].dtype == result_dtype,  \
        "'betweenness_centrality' column has not the expected type"
    nx_bc = nx.betweenness_centrality(Gnx,
                                      k=k,
                                      normalized=normalized,
                                      weight=weight,
                                      endpoints=endpoints)

    sorted_df = df.sort_values("vertex").rename({"betweenness_centrality":
                                                 "cu_bc"})
    _, nx_bc = zip(*sorted(nx_bc.items()))
    nx_df = cudf.DataFrame({"ref_bc": nx_bc})

    merged_sorted_df = cudf.concat([sorted_df, nx_df], axis=1, sort=False)

    return merged_sorted_df


# =============================================================================
# Utils
# =============================================================================
# NOTE: We assume that both column are ordered in such way that values
#        at ith positions are expected to be compared in both columns
# i.e: sorted_df[idx][first_key] should be compared to
#      sorted_df[idx][second_key]
def compare_scores(sorted_df, first_key, second_key, epsilon=DEFAULT_EPSILON):
    errors = sorted_df[~cupy.isclose(sorted_df[first_key],
                                     sorted_df[second_key],
                                     rtol=epsilon)]
    num_errors = len(errors)
    if num_errors > 0:
        print(errors)
    assert num_errors == 0, \
        "Mismatch were found when comparing '{}' and '{}' (rtol = {})" \
        .format(first_key, second_key, epsilon)


def prepare_test():
    gc.collect()


@pytest.fixture
def use_opg_batch():
    print("[DBG][OPG] Using OPG")
    cluster = dask_cuda.LocalCUDACluster()
    client = dask.distributed.Client()
    print("[DBG][OPG] Using cluster", cluster)
    yield (cluster, client)
    client.close()
    cluster.close()

# =============================================================================
# Tests
# =============================================================================
@pytest.mark.parametrize('graph_file', DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('subset_size', SUBSET_SIZE_OPTIONS)
@pytest.mark.parametrize('normalized', NORMALIZED_OPTIONS)
@pytest.mark.parametrize('weight', [None])
@pytest.mark.parametrize('endpoints', ENDPOINTS_OPTIONS)
@pytest.mark.parametrize('subset_seed', SUBSET_SEED_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
def test_betweenness_centrality(graph_file,
                                directed,
                                subset_size,
                                normalized,
                                weight,
                                endpoints,
                                subset_seed,
                                result_dtype):
    prepare_test()
    sorted_df = calc_betweenness_centrality(graph_file,
                                            directed=directed,
                                            normalized=normalized,
                                            k=subset_size,
                                            weight=weight,
                                            endpoints=endpoints,
                                            seed=subset_seed,
                                            result_dtype=result_dtype)
    compare_scores(sorted_df, first_key="cu_bc", second_key="ref_bc")


def test_opg_betweenness_centrality():
    #rmm.reinitialize(managed_memory=True)
    directed = True
    graph_file = '../datasets/karate.csv'
    cluster = dask_cuda.LocalCUDACluster()
    client = dask.distributed.Client(cluster)
    print(cluster)
    G, Gnx = utils.build_cu_and_nx_graphs(graph_file, directed=directed)
    sorted_df = calc_betweenness_centrality(graph_file,
                                            directed=directed, result_dtype=np.float32)

    compare_scores(sorted_df, first_key="cu_bc", second_key="ref_bc")

    client.close()
    cluster.close()


@pytest.mark.parametrize('graph_file', DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('subset_size', [None])
@pytest.mark.parametrize('normalized', NORMALIZED_OPTIONS)
@pytest.mark.parametrize('weight', [None])
@pytest.mark.parametrize('endpoints', ENDPOINTS_OPTIONS)
@pytest.mark.parametrize('subset_seed', SUBSET_SEED_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
@pytest.mark.parametrize('use_k_full', [True])
def test_betweenness_centrality_k_full(graph_file,
                                       directed,
                                       subset_size,
                                       normalized,
                                       weight,
                                       endpoints,
                                       subset_seed,
                                       result_dtype,
                                       use_k_full):
    """Tests full betweenness centrality by using k = G.number_of_vertices()
    instead of k=None, checks that k scales properly"""
    prepare_test()
    sorted_df = calc_betweenness_centrality(graph_file,
                                            directed=directed,
                                            normalized=normalized,
                                            k=subset_size,
                                            weight=weight,
                                            endpoints=endpoints,
                                            seed=subset_seed,
                                            result_dtype=result_dtype,
                                            use_k_full=use_k_full)
    compare_scores(sorted_df, first_key="cu_bc", second_key="ref_bc")


# NOTE: This test should only be execute on unrenumbered datasets
#       the function operating the comparison inside is first proceeding
#       to a random sampling over the number of vertices (thus direct offsets)
#       in the graph structure instead of actual vertices identifiers
@pytest.mark.parametrize('graph_file', UNRENUMBERED_DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('subset_size', SUBSET_SIZE_OPTIONS)
@pytest.mark.parametrize('normalized', NORMALIZED_OPTIONS)
@pytest.mark.parametrize('weight', [None])
@pytest.mark.parametrize('endpoints', ENDPOINTS_OPTIONS)
@pytest.mark.parametrize('subset_seed', [None])
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
def test_betweenness_centrality_fixed_sample(graph_file,
                                             directed,
                                             subset_size,
                                             normalized,
                                             weight,
                                             endpoints,
                                             subset_seed,
                                             result_dtype):
    """Test Betweenness Centrality using a subset

    Only k sources are considered for an approximate Betweenness Centrality
    """
    prepare_test()
    sorted_df = calc_betweenness_centrality(graph_file,
                                            directed=directed,
                                            k=subset_size,
                                            normalized=normalized,
                                            weight=weight,
                                            endpoints=endpoints,
                                            seed=subset_seed,
                                            result_dtype=result_dtype)
    compare_scores(sorted_df, first_key="cu_bc", second_key="ref_bc")


@pytest.mark.parametrize('graph_file', DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('subset_size', SUBSET_SIZE_OPTIONS)
@pytest.mark.parametrize('normalized', NORMALIZED_OPTIONS)
@pytest.mark.parametrize('weight', [[]])
@pytest.mark.parametrize('endpoints', ENDPOINTS_OPTIONS)
@pytest.mark.parametrize('subset_seed', SUBSET_SEED_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
def test_betweenness_centrality_weight_except(graph_file,
                                              directed,
                                              subset_size,
                                              normalized,
                                              weight,
                                              endpoints,
                                              subset_seed,
                                              result_dtype):
    """Calls betwenness_centrality with weight

    As of 05/28/2020, weight is not supported and should raise
    a NotImplementedError
    """
    prepare_test()
    with pytest.raises(NotImplementedError):
        sorted_df = calc_betweenness_centrality(graph_file,
                                                directed=directed,
                                                k=subset_size,
                                                normalized=normalized,
                                                weight=weight,
                                                endpoints=endpoints,
                                                seed=subset_seed,
                                                result_dtype=result_dtype)
        compare_scores(sorted_df, first_key="cu_bc", second_key="ref_bc")


@pytest.mark.parametrize('graph_file', DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('normalized', NORMALIZED_OPTIONS)
@pytest.mark.parametrize('subset_size', SUBSET_SIZE_OPTIONS)
@pytest.mark.parametrize('weight', [None])
@pytest.mark.parametrize('endpoints', ENDPOINTS_OPTIONS)
@pytest.mark.parametrize('subset_seed', SUBSET_SEED_OPTIONS)
@pytest.mark.parametrize('result_dtype', [str])
def test_betweenness_invalid_dtype(graph_file,
                                   directed,
                                   subset_size,
                                   normalized,
                                   weight,
                                   endpoints,
                                   subset_seed,
                                   result_dtype):
    """Test calls edge_betwenness_centrality an invalid type"""

    prepare_test()
    with pytest.raises(TypeError):
        sorted_df = calc_betweenness_centrality(graph_file,
                                                directed=directed,
                                                k=subset_size,
                                                normalized=normalized,
                                                weight=weight,
                                                endpoints=endpoints,
                                                seed=subset_seed,
                                                result_dtype=result_dtype)
        compare_scores(sorted_df, first_key="cu_bc", second_key="ref_bc")
