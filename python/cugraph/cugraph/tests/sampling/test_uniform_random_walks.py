# Copyright (c) 2025, NVIDIA CORPORATION.:
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

import cudf
import cugraph
from cudf.testing.testing import assert_frame_equal
from cugraph.testing import SMALL_DATASETS, DEFAULT_DATASETS


# =============================================================================
# Parameters
# =============================================================================
DIRECTED_GRAPH_OPTIONS = [False, True]
WEIGHTED_GRAPH_OPTIONS = [False, True]
DATASETS = [pytest.param(d) for d in DEFAULT_DATASETS]
SMALL_DATASETS = [pytest.param(d) for d in SMALL_DATASETS]


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def calc_uniform_random_walks(G, max_depth=None):
    """
    compute random walks for each nodes in 'start_vertices'

    parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
        The graph can be either directed or undirected.
        Weights in the graph are ignored.
        Use weight parameter if weights need to be considered
        (currently not supported)

    max_depth : int
        The maximum depth of the random walks

    Returns
    -------
    vertex_paths : cudf.Series
        Series containing the vertices of edges/paths in the random walk.

    edge_weight_paths: cudf.Series
        Series containing the edge weights of edges represented by the
        returned vertex_paths

    sizes: int
        The path size in case of coalesced paths.
    """
    assert G is not None

    k = random.randint(1, 6)

    start_vertices = G.select_random_vertices(num_vertices=k)

    print("\nstart_vertices is \n", start_vertices)
    vertex_paths, edge_weights, vertex_path_sizes = cugraph.uniform_random_walks(
        G, start_vertices, max_depth
    )

    return (vertex_paths, edge_weights, vertex_path_sizes), start_vertices


def check_uniform_random_walks(G, path_data, seeds, max_depth):
    invalid_edge = 0
    invalid_seeds = 0
    invalid_edge_wgt = 0
    v_paths = path_data[0]
    e_wgt_paths = path_data[1]
    e_wgt_idx = 0

    df_G = G.input_df

    if "weight" in df_G.columns:
        df_G = df_G.rename(columns={"weight": "wgt"})

    total_depth = (max_depth) * len(seeds)

    for i in range(total_depth):
        if isinstance(seeds, cudf.DataFrame):
            vertex_1 = v_paths.iloc[[i]].reset_index(drop=True)
            vertex_2 = v_paths.iloc[[i + 1]].reset_index(drop=True)
        else:
            vertex_1, vertex_2 = v_paths.iloc[i], v_paths.iloc[i + 1]

        # Every max_depth'th vertex in 'v_paths' is a seed instead of
        # 'seeds[i // (max_depth + 1)]', could have just pop the first element
        # of the seeds array once there is a match and compare it to 'vertex_1'

        if i % (max_depth + 1) == 0:
            if isinstance(seeds, cudf.DataFrame):
                assert_frame_equal(
                    vertex_1.rename(
                        columns={
                            x: y
                            for x, y in zip(
                                vertex_1.columns, range(0, len(vertex_1.columns))
                            )
                        }
                    ),
                    seeds.iloc[[i // (max_depth + 1)]]
                    .reset_index(drop=True)
                    .rename(
                        columns={
                            x: y
                            for x, y in zip(seeds.columns, range(0, len(seeds.columns)))
                        }
                    ),
                    check_dtype=False,
                    check_like=True,
                )
            else:
                if i % (max_depth + 1) == 0 and vertex_1 != seeds[i // (max_depth + 1)]:
                    invalid_seeds += 1
                    print(
                        "[ERR] Invalid seed: "
                        " src {} != src {}".format(
                            vertex_1, seeds[i // (max_depth + 1)]
                        )
                    )

        if (i % (max_depth + 1)) != (max_depth):
            # These are the edges
            src = vertex_1
            dst = vertex_2

            # check for valid edge.
            if isinstance(seeds, cudf.DataFrame):
                if (-1 not in src.iloc[0].reset_index(drop=True)) and (
                    -1 not in dst.iloc[0].reset_index(drop=True)
                ):
                    edge = cudf.DataFrame()
                    edge["src"] = vertex_1["0_vertex_paths"]
                    edge["src_0"] = vertex_1["1_vertex_paths"]
                    edge["dst"] = vertex_2["0_vertex_paths"]
                    edge["dst_0"] = vertex_2["1_vertex_paths"]

                    assert len(cudf.merge(df_G, edge, on=[*edge.columns])) > 0
            else:
                edge = df_G.loc[
                    (df_G["src"] == (src)) & (df_G["dst"] == (dst))
                ].reset_index(drop=True)

                if len(edge) == 0:
                    print(
                        "[ERR] Invalid edge: "
                        "There is no edge src {} dst {}".format(src, dst)
                    )
                    invalid_edge += 1

                else:
                    # check valid edge wgt
                    if G.is_weighted():
                        expected_wgt = edge["wgt"].iloc[0]
                        result_wgt = e_wgt_paths.iloc[e_wgt_idx]

                        if expected_wgt != result_wgt:
                            print(
                                "[ERR] Invalid edge wgt: "
                                "The edge src {} dst {} has wgt {} but got {}".format(
                                    src, dst, expected_wgt, result_wgt
                                )
                            )
                            invalid_edge_wgt += 1
                e_wgt_idx += 1

                if src != -1 and dst == -1:
                    # ensure there is no outgoing edges from 'src'
                    assert G.out_degree([src])["degree"].iloc[0] == 0

    assert invalid_seeds == 0
    assert invalid_edge == 0
    assert len(v_paths) == (max_depth + 1) * len(seeds)
    if G.is_weighted():
        assert invalid_edge_wgt == 0
        assert len(e_wgt_paths) == (max_depth) * len(seeds)

    max_path_lenth = path_data[2]
    assert max_path_lenth == max_depth


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", SMALL_DATASETS)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize("max_depth", [None])
def test_uniform_random_walks_invalid_max_dept(graph_file, directed, max_depth):

    input_graph = graph_file.get_graph(create_using=cugraph.Graph(directed=directed))
    with pytest.raises(TypeError):
        _, _, _ = calc_uniform_random_walks(input_graph, max_depth=max_depth)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", SMALL_DATASETS)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_uniform_random_walks(graph_file, directed):
    max_depth = random.randint(2, 10)
    print("max_depth is ", max_depth)
    input_graph = graph_file.get_graph(create_using=cugraph.Graph(directed=directed))

    path_data, seeds = calc_uniform_random_walks(input_graph, max_depth=max_depth)

    check_uniform_random_walks(input_graph, path_data, seeds, max_depth)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", SMALL_DATASETS)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_uniform_random_walks_multi_column_seeds(graph_file, directed):
    max_depth = random.randint(2, 10)
    df_G = graph_file.get_edgelist()
    df_G.rename(columns={"wgt": "weight"}, inplace=True)
    df_G["src_0"] = df_G["src"] + 1000
    df_G["dst_0"] = df_G["dst"] + 1000

    if directed:
        G = cugraph.Graph(directed=True)
    else:
        G = cugraph.Graph()
    G.from_cudf_edgelist(
        df_G, source=["src", "src_0"], destination=["dst", "dst_0"], edge_attr="weight"
    )

    k = random.randint(1, 10)

    seeds = G.select_random_vertices(num_vertices=k)
    vertex_paths, edge_weights, vertex_path_sizes = cugraph.uniform_random_walks(
        G, seeds, max_depth
    )

    path_data = (vertex_paths, edge_weights, vertex_path_sizes)

    check_uniform_random_walks(G, path_data, seeds, max_depth)
