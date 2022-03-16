# Copyright (c) 2022, NVIDIA CORPORATION.
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
import random

import pytest

from cugraph.tests import utils
import cugraph


# =============================================================================
# Parameters
# =============================================================================
DIRECTED_GRAPH_OPTIONS = [False, True]
DATASETS_SMALL = [pytest.param(d) for d in utils.DATASETS_SMALL]
KARATE = DATASETS_SMALL[0][0][0]


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def calc_node2vec(G,
                  start_vertices,
                  max_depth=None,
                  use_padding=False,
                  p=1.0,
                  q=1.0):
    """
    Compute node2vec for each nodes in 'start_vertices'

    Parameters
    ----------
    G : cuGraph.Graph or networkx.Graph

    start_vertices : int or list or cudf.Series

    max_depth : int

    use_padding : bool

    p : float

    q : float
    """
    assert G is not None

    vertex_paths, edge_weights, vertex_path_sizes = cugraph.node2vec(
        G, start_vertices, max_depth, use_padding, p, q)
    return (vertex_paths, edge_weights, vertex_path_sizes), start_vertices


@pytest.mark.parametrize("graph_file", utils.DATASETS_SMALL)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_node2vec_coalesced(
    graph_file,
    directed
):
    G = utils.generate_cugraph_graph_from_file(graph_file, directed=directed,
                                               edgevals=True)
    k = random.randint(1, 10)
    max_depth = 3
    start_vertices = random.sample(range(G.number_of_vertices()), k)
    df, seeds = calc_node2vec(
        G,
        start_vertices,
        max_depth,
        use_padding=False,
        p=0.8,
        q=0.5
    )
    vertex_paths, edge_weights, vertex_path_sizes = df
    # Check that output sizes are as expected
    assert vertex_paths.size == max_depth * k
    assert edge_weights.size == (max_depth - 1) * k
    # Check that weights match up with paths
    err = 0
    for i in range(k):
        for j in range(max_depth - 1):
            # weight = edge_weights[i * (max_depth - 1) + j]
            u = vertex_paths[i * max_depth + j]
            v = vertex_paths[i * max_depth + j + 1]
            # Walk not found in edgelist
            if (not G.has_edge(u, v)):
                err += 1
            # FIXME: Checking weights is buggy
            # Corresponding weight to edge is not correct
            # expr = "(src == {} and dst == {})".format(u, v)
            # if not (G.edgelist.edgelist_df.query(expr)["weights"] == weight):
            #    err += 1
    assert err == 0


@pytest.mark.parametrize("graph_file", utils.DATASETS_SMALL)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_node2vec_padded(
    graph_file,
    directed
):
    G = utils.generate_cugraph_graph_from_file(graph_file, directed=directed,
                                               edgevals=True)
    k = random.randint(1, 10)
    max_depth = 3
    start_vertices = random.sample(range(G.number_of_vertices()), k)
    df, seeds = calc_node2vec(
        G,
        start_vertices,
        max_depth,
        use_padding=True,
        p=0.8,
        q=0.5
    )
    vertex_paths, edge_weights, vertex_path_sizes = df
    # Check that output sizes are as expected
    assert vertex_paths.size == max_depth * k
    assert edge_weights.size == (max_depth - 1) * k
    assert vertex_path_sizes.sum() == vertex_paths.size
    # Check that weights match up with paths
    err = 0
    path_start = 0
    for i in range(k):
        for j in range(max_depth - 1):
            # weight = edge_weights[i * (max_depth - 1) + j]
            u = vertex_paths[i * max_depth + j]
            v = vertex_paths[i * max_depth + j + 1]
            # Walk not found in edgelist
            if (not G.has_edge(u, v)):
                err += 1
            # FIXME: Checking weights is buggy
            # Corresponding weight to edge is not correct
            # expr = "(src == {} and dst == {})".format(u, v)
            # if not (G.edgelist.edgelist_df.query(expr)["weights"] == weight):
            #    err += 1
        # Check that path sizes matches up correctly with paths
        if vertex_paths[i * max_depth] != seeds[i]:
            err += 1
        path_start += vertex_path_sizes[i]
    assert err == 0


@pytest.mark.parametrize("graph_file", [KARATE])
def test_node2vec_invalid(
    graph_file
):
    G = utils.generate_cugraph_graph_from_file(graph_file, directed=True,
                                               edgevals=True)
    k = random.randint(1, 10)
    start_vertices = random.sample(range(G.number_of_vertices()), k)
    use_padding = True
    max_depth = 1
    p = 1
    q = 1
    invalid_max_depths = [None, -1, "1", 4.5]
    invalid_pqs = [None, -1, "1"]

    # Tests for invalid max_depth
    for bad_depth in invalid_max_depths:
        with pytest.raises(ValueError):
            df, seeds = calc_node2vec(G, start_vertices, max_depth=bad_depth,
                                      use_padding=use_padding, p=p, q=q)
    # Tests for invalid p
    for bad_p in invalid_pqs:
        with pytest.raises(ValueError):
            df, seeds = calc_node2vec(G, start_vertices, max_depth=max_depth,
                                      use_padding=use_padding, p=bad_p, q=q)
    # Tests for invalid q
    for bad_q in invalid_pqs:
        with pytest.raises(ValueError):
            df, seeds = calc_node2vec(G, start_vertices, max_depth=max_depth,
                                      use_padding=use_padding, p=p, q=bad_q)
