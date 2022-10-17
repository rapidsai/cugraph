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

from cugraph.testing import utils
import cugraph
import cudf
from cugraph.experimental.datasets import small_line, karate, DATASETS_SMALL


# =============================================================================
# Parameters
# =============================================================================
DIRECTED_GRAPH_OPTIONS = [False, True]
COMPRESSED = [False, True]
LINE = small_line
KARATE = karate


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def _get_param_args(param_name, param_values):
    """
    Returns a tuple of (<param_name>, <pytest.param list>) which can be applied
    as the args to pytest.mark.parametrize(). The pytest.param list also
    contains param id string formed from the param name and values.
    """
    return (param_name, [pytest.param(v, id=f"{param_name}={v}") for v in param_values])


def calc_node2vec(G, start_vertices, max_depth, compress_result, p=1.0, q=1.0):
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
        G, start_vertices, max_depth, compress_result, p, q
    )
    return (vertex_paths, edge_weights, vertex_path_sizes), start_vertices


@pytest.mark.parametrize(*_get_param_args("graph_file", [KARATE]))
def test_node2vec_invalid(graph_file):
    G = graph_file.get_graph(create_using=cugraph.Graph(directed=True))
    k = random.randint(1, 10)
    start_vertices = cudf.Series(
        random.sample(range(G.number_of_vertices()), k), dtype="int32"
    )
    compress = True
    max_depth = 1
    p = 1
    q = 1
    invalid_max_depths = [None, -1, "1", 4.5]
    invalid_pqs = [None, -1, "1"]
    invalid_start_vertices = [1.0, "1", 2147483648]

    # Tests for invalid max_depth
    for bad_depth in invalid_max_depths:
        with pytest.raises(ValueError):
            df, seeds = calc_node2vec(
                G,
                start_vertices,
                max_depth=bad_depth,
                compress_result=compress,
                p=p,
                q=q,
            )
    # Tests for invalid p
    for bad_p in invalid_pqs:
        with pytest.raises(ValueError):
            df, seeds = calc_node2vec(
                G,
                start_vertices,
                max_depth=max_depth,
                compress_result=compress,
                p=bad_p,
                q=q,
            )
    # Tests for invalid q
    for bad_q in invalid_pqs:
        with pytest.raises(ValueError):
            df, seeds = calc_node2vec(
                G,
                start_vertices,
                max_depth=max_depth,
                compress_result=compress,
                p=p,
                q=bad_q,
            )

    # Tests for invalid start_vertices dtypes, modify when more types are
    # supported
    for bad_start in invalid_start_vertices:
        with pytest.raises(ValueError):
            df, seeds = calc_node2vec(
                G, bad_start, max_depth=max_depth, compress_result=compress, p=p, q=q
            )


@pytest.mark.parametrize(*_get_param_args("graph_file", [LINE]))
@pytest.mark.parametrize(*_get_param_args("directed", DIRECTED_GRAPH_OPTIONS))
def test_node2vec_line(graph_file, directed):
    G = graph_file.get_graph(create_using=cugraph.Graph(directed=directed))
    max_depth = 3
    start_vertices = cudf.Series([0, 3, 6], dtype="int32")
    df, seeds = calc_node2vec(
        G, start_vertices, max_depth, compress_result=True, p=0.8, q=0.5
    )


@pytest.mark.parametrize(*_get_param_args("graph_file", DATASETS_SMALL))
@pytest.mark.parametrize(*_get_param_args("directed", DIRECTED_GRAPH_OPTIONS))
@pytest.mark.parametrize(*_get_param_args("compress", COMPRESSED))
def test_node2vec(
    graph_file,
    directed,
    compress,
):
    dataset_path = graph_file.get_path()
    cu_M = utils.read_csv_file(dataset_path)

    G = cugraph.Graph(directed=directed)

    G.from_cudf_edgelist(
        cu_M, source="0", destination="1", edge_attr="2", renumber=False
    )
    num_verts = G.number_of_vertices()
    k = random.randint(6, 12)
    start_vertices = cudf.Series(random.sample(range(num_verts), k), dtype="int32")
    max_depth = 5
    result, seeds = calc_node2vec(
        G, start_vertices, max_depth, compress_result=compress, p=0.8, q=0.5
    )
    vertex_paths, edge_weights, vertex_path_sizes = result

    if compress:
        # Paths are coalesced, meaning vertex_path_sizes is nonempty. It's
        # necessary to use in order to track starts of paths
        assert vertex_paths.size == vertex_path_sizes.sum()
        if directed:
            # directed graphs may be coalesced at any point
            assert vertex_paths.size - k == edge_weights.size
            # This part is for checking to make sure each of the edges
            # in all of the paths are valid and are accurate
            idx = 0
            for path_idx in range(vertex_path_sizes.size):
                for _ in range(vertex_path_sizes[path_idx] - 1):
                    weight = edge_weights[idx]
                    u = vertex_paths[idx + path_idx]
                    v = vertex_paths[idx + path_idx + 1]
                    # Corresponding weight to edge is not correct
                    expr = "(src == {} and dst == {})".format(u, v)
                    edge_query = G.edgelist.edgelist_df.query(expr)
                    if edge_query.empty:
                        raise ValueError("edge_query didn't find:({},{})".format(u, v))
                    else:
                        if edge_query["weights"].values[0] != weight:
                            raise ValueError("edge_query weight incorrect")
                    idx += 1

        else:
            # undirected graphs should never be coalesced
            assert vertex_paths.size == max_depth * k
            assert edge_weights.size == (max_depth - 1) * k
            # This part is for checking to make sure each of the edges
            # in all of the paths are valid and are accurate
            for path_idx in range(k):
                for idx in range(max_depth - 1):
                    weight = edge_weights[path_idx * (max_depth - 1) + idx]
                    u = vertex_paths[path_idx * max_depth + idx]
                    v = vertex_paths[path_idx * max_depth + idx + 1]
                    # Corresponding weight to edge is not correct
                    expr = "(src == {} and dst == {})".format(u, v)
                    edge_query = G.edgelist.edgelist_df.query(expr)
                    if edge_query.empty:
                        raise ValueError("edge_query didn't find:({},{})".format(u, v))
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
                # Corresponding weight to edge is not correct
                expr = "(src == {} and dst == {})".format(u, v)
                edge_query = G.edgelist.edgelist_df.query(expr)
                if edge_query.empty:
                    raise ValueError("edge_query didn't find:({},{})".format(u, v))
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
                raise ValueError(
                    "vertex_path start did not match seed \
                                 vertex:{}".format(
                        vertex_paths.values
                    )
                )


@pytest.mark.parametrize(*_get_param_args("graph_file", [LINE]))
@pytest.mark.parametrize(*_get_param_args("renumber", [True, False]))
def test_node2vec_renumber_cudf(graph_file, renumber):
    dataset_path = graph_file.get_path()
    cu_M = cudf.read_csv(
        dataset_path, delimiter=" ", dtype=["int32", "int32", "float32"], header=None
    )
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(
        cu_M, source="0", destination="1", edge_attr="2", renumber=renumber
    )

    start_vertices = cudf.Series([8, 0, 7, 1, 6, 2], dtype="int32")
    num_seeds = 6
    max_depth = 4

    df, seeds = calc_node2vec(
        G, start_vertices, max_depth, compress_result=False, p=0.8, q=0.5
    )
    vertex_paths, edge_weights, vertex_path_sizes = df

    for i in range(num_seeds):
        if vertex_paths[i * max_depth] != seeds[i]:
            raise ValueError(
                "vertex_path {} start did not match seed \
                             vertex".format(
                    vertex_paths.values
                )
            )
