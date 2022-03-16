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
# KARATE = DATASETS_SMALL[0][0][0]
KARATE = utils.RAPIDS_DATASET_ROOT_DIR_PATH/"karate.csv"
ARROW = utils.RAPIDS_DATASET_ROOT_DIR_PATH/"small_arrow.csv"


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
    for i in range(k):
        for j in range(max_depth - 1):
            weight = edge_weights[i * (max_depth - 1) + j]
            u = vertex_paths[i * max_depth + j]
            v = vertex_paths[i * max_depth + j + 1]
            # Walk not found in edgelist
            edge_found = G.has_edge(u, v)
            if not edge_found:
                raise ValueError("Edge {},{} not found".format(u, v))
            # FIXME: Checking weights is buggy
            # Corresponding weight to edge is not correct
            expr = "(src == {} and dst == {})".format(u, v)
            edge_query = G.edgelist.edgelist_df.query(expr)
            if edge_query.empty:
                raise ValueError("edge_query yielded no edge")
            else:
                if edge_query["weights"].values[0] != weight:
                    raise ValueError("edge_query weight incorrect")


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
    path_start = 0
    for i in range(k):
        for j in range(max_depth - 1):
            weight = edge_weights[i * (max_depth - 1) + j]
            u = vertex_paths[i * max_depth + j]
            v = vertex_paths[i * max_depth + j + 1]
            # Walk not found in edgelist
            edge_found = G.has_edge(u, v)
            if not edge_found:
                raise ValueError("Edge {},{} not found".format(u, v))
            # FIXME: Checking weights is buggy
            # Corresponding weight to edge is not correct
            expr = "(src == {} and dst == {})".format(u, v)
            edge_query = G.edgelist.edgelist_df.query(expr)
            if edge_query.empty:
                raise ValueError("edge_query yielded no edge")
            else:
                if edge_query["weights"].values[0] != weight:
                    raise ValueError("edge_query weight incorrect")
        # Check that path sizes matches up correctly with paths
        if vertex_paths[i * max_depth] != seeds[i]:
            raise ValueError("vertex_path start did not match seed vertex")
        path_start += vertex_path_sizes[i]


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


@pytest.mark.parametrize("graph_file", [ARROW])
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_node2vec_arrow(graph_file, directed):
    G = utils.generate_cugraph_graph_from_file(graph_file, directed=directed,
                                               edgevals=True)
    max_depth = 3
    start_vertices = [0, 3, 6]
    df, seeds = calc_node2vec(
        G,
        start_vertices,
        max_depth,
        use_padding=True,
        p=0.8,
        q=0.5
    )


# NOTE: For this test, a custom dataset csv was created, called
# small_arrow.csv. It consists of 9 edges, src vertices 0-8 with
# corresponding dst vertices 1-9. All edge weights are 1.0.
@pytest.mark.parametrize("graph_file", [ARROW])
@pytest.mark.parametrize("renumbered", [True, False])
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_node2vec_arrow_renumbered(graph_file, renumbered, directed):
    """
    Because of how arrow csv file works, node2vec output depends on whether
    graph is directed or not.

    If directed, then the vertex paths must be 0, 1, 2, 3, 4, 5, 6, 7, 8 as
    the sampling is limited to only 1 option

    If undirected, then the vertex paths must be 0, 1, a, 3, b, c, 6, d, e,
    where 'a' = [0, 2], 'b' = [2, 4], 'c' = [1, 3, 5], 'd' = [5, 7], and
    'e' = [4, 6, 8].
    """
    from cudf import read_csv
    M = read_csv(graph_file, delimiter=' ',
                 dtype=['int32', 'int32', 'float32'], header=None)
    G = cugraph.Graph(directed=directed)
    G.from_cudf_edgelist(M, source='0', destination='1', edge_attr='2',
                         renumber=renumbered)
    max_depth = 3
    start_vertices = [0, 3, 6]
    k = len(start_vertices)
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
    # The sampling when graph is directed should be deterministic
    if directed:
        index = 0
        for vertex in vertex_paths.values:
            if vertex != index:
                raise ValueError("Directed path should be monotonic inc.")
    # To ensure renumbering works as intended, verify the starting vertex
    # is the same as the seed vertices
    for i in range(k):
        if start_vertices[i] != vertex_paths[i * max_depth]:
            raise ValueError("Starting vertex not the same as seed vertex")


# This was an attempt at creating a custom dataset from cudf, without resorting
# to creating a new csv file, unfortunately this wasn't possible as of yet.
# A possible new test in the future, though.
"""
@pytest.mark.parametrize("store_transposed", [True, False])
@pytest.mark.parametrize("renumbered", [True, False])
@pytest.mark.parametrize("do_expensive_check", [True, False])
def test_node2vec_renumbered(store_transposed, renumbered, do_expensive_check):
    # from cudf import DataFrame
    import cudf, numpy
    ex_graph = cudf.DataFrame(data=[(0, 1, 1), (1, 2, 1), (2, 3, 1),
                                    (3, 4, 1), (4, 5, 1), (5, 6, 1),
                                    (6, 7, 1), (7, 8, 1), (8, 9, 1)],
                                    columns=["0", "1", "2"],
                                    dtype=numpy.int32)
    #ex_graph = cudf.DataFrame({"0": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    #         "1": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    #         "2": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]})

    g = cugraph.Graph()
    g.from_cudf_edgelist(ex_graph, source="0", destination="1", edge_attr="2",
                          renumber=renumbered)
    max_depth = 3
    start_vertices = cudf.Series([0, 3, 6], dtype=numpy.int32)

    cugraph.node2vec(g, start_vertices, max_depth,
                     use_padding=False, p=0.8, q=0.5)
"""
