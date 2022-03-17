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
# Uncomment when bug is resolved
# KARATE = DATASETS_SMALL[0][0][0]
# Temporary for bug squashing
KARATE = utils.RAPIDS_DATASET_ROOT_DIR_PATH/"karate.csv"
LINE = utils.RAPIDS_DATASET_ROOT_DIR_PATH/"small_line.csv"


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def calc_node2vec(G,
                  start_vertices,
                  max_depth=None,
                  compress_result=False,
                  p=1.0,
                  q=1.0):
    """
    Compute node2vec for each nodes in 'start_vertices'

    Parameters
    ----------
    G : cuGraph.Graph or networkx.Graph

    start_vertices : int or list or cudf.Series

    max_depth : int

    compress_result : bool

    p : float

    q : float
    """
    assert G is not None

    vertex_paths, edge_weights, vertex_path_sizes = cugraph.node2vec(
        G, start_vertices, max_depth, compress_result, p, q)
    return (vertex_paths, edge_weights, vertex_path_sizes), start_vertices


@pytest.mark.parametrize("graph_file", [KARATE])
def test_node2vec_invalid(
    graph_file
):
    G = utils.generate_cugraph_graph_from_file(graph_file, directed=True,
                                               edgevals=True)
    k = random.randint(1, 10)
    start_vertices = random.sample(range(G.number_of_vertices()), k)
    compress = True
    max_depth = 1
    p = 1
    q = 1
    invalid_max_depths = [None, -1, "1", 4.5]
    invalid_pqs = [None, -1, "1"]

    # Tests for invalid max_depth
    for bad_depth in invalid_max_depths:
        with pytest.raises(ValueError):
            df, seeds = calc_node2vec(G, start_vertices, max_depth=bad_depth,
                                      compress_result=compress, p=p, q=q)
    # Tests for invalid p
    for bad_p in invalid_pqs:
        with pytest.raises(ValueError):
            df, seeds = calc_node2vec(G, start_vertices, max_depth=max_depth,
                                      compress_result=compress, p=bad_p, q=q)
    # Tests for invalid q
    for bad_q in invalid_pqs:
        with pytest.raises(ValueError):
            df, seeds = calc_node2vec(G, start_vertices, max_depth=max_depth,
                                      compress_result=compress, p=p, q=bad_q)


@pytest.mark.parametrize("graph_file", [LINE])
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_node2vec_line(graph_file, directed):
    G = utils.generate_cugraph_graph_from_file(graph_file, directed=directed,
                                               edgevals=True)
    max_depth = 3
    start_vertices = [0, 3, 6]
    df, seeds = calc_node2vec(
        G,
        start_vertices,
        max_depth,
        compress_result=True,
        p=0.8,
        q=0.5
    )


# NOTE: For this test, a custom dataset csv was created, called
# small_line.csv. It consists of 9 edges, src vertices 0-8 with
# corresponding dst vertices 1-9. All edge weights are 1.0.
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


@pytest.mark.parametrize("graph_file", [KARATE, LINE])
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize("compress", [True, False])
def test_node2vec_new(
    graph_file,
    directed,
    compress
):
    G = utils.generate_cugraph_graph_from_file(graph_file, directed=directed,
                                               edgevals=True)
    num_verts = G.number_of_vertices()
    if graph_file == KARATE:
        k = random.randint(1, 7)
        start_vertices = random.sample(range(num_verts), k)
    else:
        k = 3
        start_vertices = [3, 6, 9]
    max_depth = 5
    df, seeds = calc_node2vec(
        G,
        start_vertices,
        max_depth,
        compress_result=compress,
        p=0.8,
        q=0.5
    )
    vertex_paths, edge_weights, vertex_path_sizes = df
    if compress:
        # Paths are coalesced, meaning vertex_path_sizes is nonempty. It's
        # necessary to use in order to track starts of paths
        assert vertex_paths.size == vertex_path_sizes.sum()
        if directed:
            # directed graphs may be coalesced at any point
            assert vertex_paths.size - k == edge_weights.size
            # This part is for checking to make sure each of the edges
            # in all of the paths are valid and are accurate
            curr = 0
            path = 0
            for i in range(vertex_path_sizes.size):
                for j in range(vertex_path_sizes[i] - 1):
                    weight = edge_weights[curr]
                    u = vertex_paths[curr + path]
                    v = vertex_paths[curr + path + 1]
                    edge_found = G.has_edge(u, v)
                    if not edge_found:
                        raise ValueError("Edge {},{} not found".format(u, v))
                    expr = "(src == {} and dst == {})".format(u, v)
                    edge_query = G.edgelist.edgelist_df.query(expr)
                    if edge_query.empty:
                        raise ValueError("edge_query didn't find:({},{}),{}".
                                         format(u, v, num_verts))
                    else:
                        if edge_query["weights"].values[0] != weight:
                            raise ValueError("edge_query weight incorrect")
                    curr += 1
                path += 1
        else:
            # undirected graphs should never be coalesced
            assert vertex_paths.size == max_depth * k
            assert edge_weights.size == (max_depth - 1) * k
            # This part is for checking to make sure each of the edges
            # in all of the paths are valid and are accurate
            for i in range(k):
                for j in range(max_depth - 1):
                    weight = edge_weights[i * (max_depth - 1) + j]
                    u = vertex_paths[i * max_depth + j]
                    v = vertex_paths[i * max_depth + j + 1]
                    # Walk not found in edgelist
                    edge_found = G.has_edge(u, v)
                    if not edge_found:
                        raise ValueError("Edge {},{} not found".format(u, v))
                    # Corresponding weight to edge is not correct
                    expr = "(src == {} and dst == {})".format(u, v)
                    edge_query = G.edgelist.edgelist_df.query(expr)
                    if edge_query.empty:
                        raise ValueError("edge_query didn't find:({},{}),{}".
                                         format(u, v, num_verts))
                    else:
                        if edge_query["weights"].values[0] != weight:
                            raise ValueError("edge_query weight incorrect")
    else:
        # Paths are padded, meaning a formula can be used to track starts of
        # paths. Check that output sizes are as expected
        assert vertex_paths.size == max_depth * k
        assert edge_weights.size == (max_depth - 1) * k
        assert vertex_path_sizes.size == 0
        if directed:
            blanks = vertex_paths.isna()
        # This part is for checking to make sure each of the edges
        # in all of the paths are valid and are accurate
        for i in range(k):
            path_at_end, j = False, 0
            weight_idx = 0
            while not path_at_end:
                src_idx = i * max_depth + j
                dst_idx = i * max_depth + j + 1
                if directed:
                    invalid_src = blanks[src_idx] or (src_idx >= num_verts)
                    invalid_dst = blanks[dst_idx] or (dst_idx >= num_verts)
                    if invalid_src or invalid_dst:
                        break
                weight = edge_weights[weight_idx]
                u = vertex_paths[src_idx]
                v = vertex_paths[dst_idx]
                # Walk not found in edgelist
                edge_found = G.has_edge(u, v)
                if not edge_found:
                    raise ValueError("Edge {},{} not found".format(u, v))
                # Corresponding weight to edge is not correct
                expr = "(src == {} and dst == {})".format(u, v)
                edge_query = G.edgelist.edgelist_df.query(expr)
                if edge_query.empty:
                    raise ValueError("edge_query didn't find:({},{}),{}".
                                     format(u, v, num_verts))
                else:
                    if edge_query["weights"].values[0] != weight:
                        raise ValueError("edge_query weight incorrect")

                # Only increment if the current indices are valid
                j += 1
                weight_idx += 1
                if j >= max_depth - 1:
                    path_at_end = True
            # Check that path sizes matches up correctly with paths
            if vertex_paths[i * max_depth] != seeds[i]:
                raise ValueError("vertex_path start did not match seed \
                                 vertex:{}".format(vertex_paths.values))
