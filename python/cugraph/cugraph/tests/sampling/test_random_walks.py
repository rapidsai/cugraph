# Copyright (c) 2020-2024, NVIDIA CORPORATION.:
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
import networkx as nx

import cudf
import cugraph
from cudf.testing import assert_series_equal
from cugraph.utilities import ensure_cugraph_obj_for_nx
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


def calc_random_walks(G, max_depth=None, use_padding=False, legacy_result_type=True):
    """
    compute random walks for each nodes in 'start_vertices'

    parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
        The graph can be either directed or undirected.
        Weights in the graph are ignored.
        Use weight parameter if weights need to be considered
        (currently not supported)

    start_vertices : int or list or cudf.Series
        A single node or a list or a cudf.Series of nodes from which to run
        the random walks

    max_depth : int
        The maximum depth of the random walks

    use_padding : bool
        If True, padded paths are returned else coalesced paths are returned.

    Returns
    -------
    vertex_paths : cudf.Series or cudf.DataFrame
        Series containing the vertices of edges/paths in the random walk.

    edge_weight_paths: cudf.Series
        Series containing the edge weights of edges represented by the
        returned vertex_paths

    sizes: int
        The path size in case of coalesced paths.
    """
    assert G is not None

    G, _ = ensure_cugraph_obj_for_nx(G, nx_weight_attr="wgt")

    k = random.randint(1, 6)

    random_walks_type = "uniform"

    start_vertices = G.select_random_vertices(num_vertices=k)

    print("\nstart_vertices is \n", start_vertices)
    vertex_paths, edge_weights, vertex_path_sizes = cugraph.random_walks(
        G, random_walks_type, start_vertices, max_depth, use_padding, legacy_result_type
    )

    return (vertex_paths, edge_weights, vertex_path_sizes), start_vertices


def check_random_walks(path_data, seeds, G):
    invalid_edge = 0
    invalid_seeds = 0
    offsets_idx = 0
    next_path_idx = 0
    v_paths = path_data[0]
    df_G = G.input_df

    sizes = path_data[2].to_numpy().tolist()

    for s in sizes:
        for i in range(next_path_idx, next_path_idx + s - 1):
            src, dst = v_paths.iloc[i], v_paths.iloc[i + 1]
            if i == next_path_idx and src != seeds[offsets_idx]:
                invalid_seeds += 1
                print(
                    "[ERR] Invalid seed: "
                    " src {} != src {}".format(src, seeds[offsets_idx])
                )
        offsets_idx += 1
        next_path_idx += s

        exp_edge = df_G.loc[
            (df_G["src"] == (src)) & (df_G["dst"] == (dst))
        ].reset_index(drop=True)

        if len(exp_edge) == 0:
            print(
                "[ERR] Invalid edge: " "There is no edge src {} dst {}".format(src, dst)
            )
            invalid_edge += 1

    assert invalid_edge == 0
    assert invalid_seeds == 0


def check_random_walks_padded(G, path_data, seeds, max_depth, legacy_result_type=True):
    invalid_edge = 0
    invalid_seeds = 0
    invalid_edge_wgt = 0
    v_paths = path_data[0]
    e_wgt_paths = path_data[1]
    e_wgt_idx = 0

    G, _ = ensure_cugraph_obj_for_nx(G, nx_weight_attr="wgt")
    df_G = G.input_df

    if "weight" in df_G.columns:
        df_G = df_G.rename(columns={"weight": "wgt"})

    total_depth = (max_depth) * len(seeds)

    for i in range(total_depth - 1):
        vertex_1, vertex_2 = v_paths.iloc[i], v_paths.iloc[i + 1]

        # Every max_depth'th vertex in 'v_paths' is a seed
        # instead of 'seeds[i // (max_depth)]', could have just pop the first element
        # of the seeds array once there is a match and compare it to 'vertex_1'
        if i % (max_depth) == 0 and vertex_1 != seeds[i // (max_depth)]:
            invalid_seeds += 1
            print(
                "[ERR] Invalid seed: "
                " src {} != src {}".format(vertex_1, seeds[i // (max_depth)])
            )

        if (i % (max_depth)) != (max_depth - 1):
            # These are the edges
            src = vertex_1
            dst = vertex_2

            if src != -1 and dst != -1:
                # check for valid edge.
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
    assert len(v_paths) == (max_depth) * len(seeds)
    if G.is_weighted():
        assert invalid_edge_wgt == 0
        assert len(e_wgt_paths) == (max_depth - 1) * len(seeds)

    if legacy_result_type:
        sizes = path_data[2]
        assert sizes is None
    else:
        max_path_lenth = path_data[2]
        assert max_path_lenth == max_depth - 1


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", SMALL_DATASETS)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize("max_depth", [None])
def test_random_walks_invalid_max_dept(graph_file, directed, max_depth):

    input_graph = graph_file.get_graph(create_using=cugraph.Graph(directed=directed))
    with pytest.raises(TypeError):
        _, _, _ = calc_random_walks(input_graph, max_depth=max_depth)


@pytest.mark.sg
@pytest.mark.cugraph_ops
@pytest.mark.parametrize("graph_file", SMALL_DATASETS)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_random_walks_coalesced(graph_file, directed):
    max_depth = random.randint(2, 10)

    input_graph = graph_file.get_graph(create_using=cugraph.Graph(directed=directed))

    path_data, seeds = calc_random_walks(
        input_graph, max_depth=max_depth, use_padding=False
    )
    check_random_walks(path_data, seeds, input_graph)

    # Check path query output
    df = cugraph.rw_path(len(seeds), path_data[2])
    v_offsets = [0] + path_data[2].cumsum()[:-1].to_numpy().tolist()
    w_offsets = [0] + (path_data[2] - 1).cumsum()[:-1].to_numpy().tolist()

    assert_series_equal(df["weight_sizes"], path_data[2] - 1, check_names=False)
    assert df["vertex_offsets"].to_numpy().tolist() == v_offsets
    assert df["weight_offsets"].to_numpy().tolist() == w_offsets


@pytest.mark.sg
@pytest.mark.cugraph_ops
@pytest.mark.parametrize("graph_file", SMALL_DATASETS)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_random_walks_padded_0(graph_file, directed):
    max_depth = random.randint(2, 10)
    print("max_depth is ", max_depth)
    input_graph = graph_file.get_graph(create_using=cugraph.Graph(directed=directed))

    path_data, seeds = calc_random_walks(
        input_graph, max_depth=max_depth, use_padding=True
    )

    check_random_walks_padded(input_graph, path_data, seeds, max_depth)

    # test for 'legacy_result_type=False'
    path_data, seeds = calc_random_walks(
        input_graph, max_depth=max_depth, use_padding=True, legacy_result_type=False
    )
    # Non 'legacy_result_type' has an extra edge 'path_data'
    check_random_walks_padded(
        input_graph, path_data, seeds, max_depth + 1, legacy_result_type=False
    )


@pytest.mark.sg
@pytest.mark.cugraph_ops
def test_random_walks_padded_1():
    max_depth = random.randint(2, 10)

    df = cudf.DataFrame()
    df["src"] = [1, 2, 4, 7, 3]
    df["dst"] = [5, 4, 1, 5, 2]
    df["wgt"] = [0.4, 0.5, 0.6, 0.7, 0.8]

    input_graph = cugraph.Graph(directed=True)

    input_graph.from_cudf_edgelist(
        df, source="src", destination="dst", edge_attr="wgt", renumber=True
    )

    path_data, seeds = calc_random_walks(
        input_graph, max_depth=max_depth, use_padding=True
    )

    check_random_walks_padded(input_graph, path_data, seeds, max_depth)


@pytest.mark.sg
@pytest.mark.cugraph_ops
@pytest.mark.parametrize("graph_file", SMALL_DATASETS)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_random_walks_nx(graph_file):
    G = graph_file.get_graph(create_using=cugraph.Graph(directed=True))

    M = G.to_pandas_edgelist()

    source = G.source_columns
    target = G.destination_columns
    edge_attr = G.weight_column

    Gnx = nx.from_pandas_edgelist(
        M,
        source=source,
        target=target,
        edge_attr=edge_attr,
        create_using=nx.DiGraph(),
    )
    max_depth = random.randint(2, 10)
    path_data, seeds = calc_random_walks(Gnx, max_depth=max_depth, use_padding=True)

    check_random_walks_padded(Gnx, path_data, seeds, max_depth)


"""@pytest.mark.parametrize("graph_file", utils.DATASETS_SMALL)
@pytest.mark.sg
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_random_walks(
    graph_file,
    directed
):
    max_depth = random.randint(2, 10)
    df_G = utils.read_csv_file(graph_file)
    df_G.rename(
        columns={"0": "src", "1": "dst", "2": "weight"}, inplace=True)
    df_G['src_0'] = df_G['src'] + 1000
    df_G['dst_0'] = df_G['dst'] + 1000

    if directed:
        G = cugraph.Graph(directed=True)
    else:
        G = cugraph.Graph()
    G.from_cudf_edgelist(df_G, source=['src', 'src_0'],
                         destination=['dst', 'dst_0'],
                         edge_attr="weight")

    k = random.randint(1, 10)
    start_vertices = random.sample(G.nodes().to_numpy().tolist(), k)

    seeds = cudf.DataFrame()
    seeds['v'] = start_vertices
    seeds['v_0'] = seeds['v'] + 1000

    df, offsets = cugraph.random_walks(G, seeds, max_depth)

    check_random_walks(df, offsets, seeds, df_G)
"""
