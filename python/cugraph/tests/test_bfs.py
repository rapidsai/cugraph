# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import cupy
import numpy as np
import pytest
import cugraph
from cugraph.tests import utils
import random

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx
    import networkx.algorithms.centrality.betweenness as nxacb

# =============================================================================
# Parameters
# =============================================================================
DIRECTED_GRAPH_OPTIONS = [True, False]

SUBSET_SEED_OPTIONS = [42]

DEFAULT_EPSILON = 1e-6


# =============================================================================
# Utils
# =============================================================================
def prepare_test():
    gc.collect()


# =============================================================================
# Functions for comparison
# =============================================================================
# NOTE: We need to use relative error, the values of the shortest path
# counters can reach extremely high values 1e+80 and above
def compare_single_sp_counter(result, expected, epsilon=DEFAULT_EPSILON):
    return np.isclose(result, expected, rtol=epsilon)


def compare_bfs(graph_file, directed=True, return_sp_counter=False, seed=42):
    """ Genereate both cugraph and reference bfs traversal

    Parameters
    -----------
    graph_file : string
        Path to COO Graph representation in .csv format

    directed : bool, optional, default=True
        Indicated whether the graph is directed or not

    return_sp_counter : bool, optional, default=False
        Retrun shortest path counters from traversal if True

    seed : int, optional, default=42
        Value for random seed to obtain starting vertex

    Returns
    -------
    """
    G, Gnx = utils.build_cu_and_nx_graphs(graph_file, directed)
    # Seed for reproducibility
    if isinstance(seed, int):
        random.seed(seed)
        start_vertex = random.sample(Gnx.nodes(), 1)[0]

        # Test for  shortest_path_counter
        compare_func = _compare_bfs_spc if return_sp_counter else _compare_bfs

        # NOTE: We need to take 2 different path for verification as the nx
        #       functions used as reference return dictionaries that might
        #       not contain all the vertices while the cugraph version return
        #       a cudf.DataFrame with all the vertices, also some verification
        #       become slow with the data transfer
        compare_func(G, Gnx, start_vertex)
    elif isinstance(seed, list):  # For other Verifications
        for start_vertex in seed:
            compare_func = (
                _compare_bfs_spc if return_sp_counter else _compare_bfs
            )
            compare_func(G, Gnx, start_vertex)
    elif seed is None:  # Same here, it is only to run full checks
        for start_vertex in Gnx:
            compare_func = (
                _compare_bfs_spc if return_sp_counter else _compare_bfs
            )
            compare_func(G, Gnx, start_vertex)
    else:  # Unknown type given to seed
        raise NotImplementedError("Invalid type for seed")


def _compare_bfs(G, Gnx, source):
    df = cugraph.bfs(G, source, return_sp_counter=False)
    # This call should only contain 3 columns:
    # 'vertex', 'distance', 'predecessor'
    # It also confirms wether or not 'sp_counter' has been created by the call
    # 'sp_counter' triggers atomic operations in BFS, thus we want to make
    # sure that it was not the case
    # NOTE: 'predecessor' is always returned while the C++ function allows to
    # pass a nullptr
    assert len(df.columns) == 3, (
        "The result of the BFS has an invalid " "number of columns"
    )
    cu_distances = {
        vertex: dist
        for vertex, dist in zip(
            df["vertex"].to_array(), df["distance"].to_array()
        )
    }
    cu_predecessors = {
        vertex: dist
        for vertex, dist in zip(
            df["vertex"].to_array(), df["predecessor"].to_array()
        )
    }

    nx_distances = nx.single_source_shortest_path_length(Gnx, source)
    # FIXME: The following only verifies vertices that were reached
    #       by cugraph's BFS.
    # We assume that the distances are given back as integers in BFS
    # max_val = np.iinfo(df['distance'].dtype).max
    # Unreached vertices have a distance of max_val

    missing_vertex_error = 0
    distance_mismatch_error = 0
    invalid_predecessor_error = 0
    for vertex in nx_distances:
        if vertex in cu_distances:
            result = cu_distances[vertex]
            expected = nx_distances[vertex]
            if result != expected:
                print(
                    "[ERR] Mismatch on distances: "
                    "vid = {}, cugraph = {}, nx = {}".format(
                        vertex, result, expected
                    )
                )
                distance_mismatch_error += 1
            if vertex not in cu_predecessors:
                missing_vertex_error += 1
            else:
                pred = cu_predecessors[vertex]
                if vertex != source and pred not in nx_distances:
                    invalid_predecessor_error += 1
                else:
                    # The graph is unweighted thus, predecessors are 1 away
                    if vertex != source and (
                        (nx_distances[pred] + 1 != cu_distances[vertex])
                    ):
                        print(
                            "[ERR] Invalid on predecessors: "
                            "vid = {}, cugraph = {}".format(vertex, pred)
                        )
                        invalid_predecessor_error += 1
        else:
            missing_vertex_error += 1
    assert missing_vertex_error == 0, "There are missing vertices"
    assert distance_mismatch_error == 0, "There are invalid distances"
    assert invalid_predecessor_error == 0, "There are invalid predecessors"


def _compare_bfs_spc(G, Gnx, source):
    df = cugraph.bfs(G, source, return_sp_counter=True)
    # This call should only contain 3 columns:
    # 'vertex', 'distance', 'predecessor', 'sp_counter'
    assert len(df.columns) == 4, (
        "The result of the BFS has an invalid " "number of columns"
    )
    _, _, nx_sp_counter = nxacb._single_source_shortest_path_basic(Gnx, source)
    sorted_nx = [nx_sp_counter[key] for key in sorted(nx_sp_counter.keys())]
    # We are not checking for distances / predecessors here as we assume
    # that these have been checked  in the _compare_bfs tests
    # We focus solely on shortest path counting

    # cugraph return a dataframe that should contain exactly one time each
    # vertex
    # We could us isin to filter only vertices that are common to both
    # But it would slow down the comparison, and in this specific case
    # nxacb._single_source_shortest_path_basic is a dictionary containing all
    # the vertices.
    # There is no guarantee when we get `df` that the vertices are sorted
    # thus we enforce the order so that we can leverage faster comparison after
    sorted_df = df.sort_values("vertex").rename({"sp_counter": "cu_spc"})

    # This will allows to detect vertices identifier that could have been
    # wrongly present multiple times
    cu_vertices = set(sorted_df["vertex"])
    nx_vertices = nx_sp_counter.keys()
    assert len(cu_vertices.intersection(nx_vertices)) == len(
        nx_vertices
    ), "There are missing vertices"

    # We add the nx shortest path counter in the cudf.DataFrame, both the
    # the DataFrame and `sorted_nx` are sorted base on vertices identifiers
    sorted_df["nx_spc"] = sorted_nx

    # We could use numpy.isclose or cupy.isclose, we can then get the entries
    # in the cudf.DataFrame where there are is a mismatch.
    # numpy / cupy allclose would get only a boolean and we might want the
    # extra information about the discrepancies
    shortest_path_counter_errors = sorted_df[
        ~cupy.isclose(
            sorted_df["cu_spc"], sorted_df["nx_spc"], rtol=DEFAULT_EPSILON
        )
    ]
    if len(shortest_path_counter_errors) > 0:
        print(shortest_path_counter_errors)
    assert len(shortest_path_counter_errors) == 0, (
        "Shortest path counters " "are too different"
    )


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.parametrize("graph_file", utils.DATASETS_5)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize("seed", SUBSET_SEED_OPTIONS)
def test_bfs(graph_file, directed, seed):
    """Test BFS traversal on random source with distance and predecessors"""
    prepare_test()
    compare_bfs(
        graph_file, directed=directed, return_sp_counter=False, seed=seed
    )


@pytest.mark.parametrize("graph_file", utils.DATASETS)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize("seed", SUBSET_SEED_OPTIONS)
def test_bfs_spc(graph_file, directed, seed):
    """Test BFS traversal on random source with shortest path counting"""
    prepare_test()
    compare_bfs(
        graph_file, directed=directed, return_sp_counter=True, seed=seed
    )


@pytest.mark.parametrize("graph_file", utils.TINY_DATASETS)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_bfs_spc_full(graph_file, directed):
    """Test BFS traversal on every vertex with shortest path counting"""
    prepare_test()
    compare_bfs(
        graph_file, directed=directed, return_sp_counter=True, seed=None
    )
