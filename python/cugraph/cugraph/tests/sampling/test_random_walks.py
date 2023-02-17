# Copyright (c) 2020-2023, NVIDIA CORPORATION.:
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
from cudf.testing import assert_series_equal

import cugraph
from cugraph.experimental.datasets import DATASETS, DATASETS_SMALL

# =============================================================================
# Parameters
# =============================================================================
DIRECTED_GRAPH_OPTIONS = [False, True]
WEIGHTED_GRAPH_OPTIONS = [False, True]
DATASETS = [pytest.param(d) for d in DATASETS]
DATASETS_SMALL = [pytest.param(d) for d in DATASETS_SMALL]


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def calc_random_walks(graph_file, directed=False, max_depth=None, use_padding=False):
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
    G = graph_file.get_graph(create_using=cugraph.Graph(directed=directed))
    assert G is not None

    k = random.randint(1, 10)
    random_walks_type = "uniform"
    start_vertices = random.sample(range(G.number_of_vertices()), k)
    vertex_paths, edge_weights, vertex_path_sizes = cugraph.random_walks(
        G, random_walks_type, start_vertices, max_depth, use_padding
    )

    return (vertex_paths, edge_weights, vertex_path_sizes), start_vertices


def check_random_walks(path_data, seeds, df_G=None):
    invalid_edge = 0
    invalid_seeds = 0
    offsets_idx = 0
    next_path_idx = 0
    v_paths = path_data[0]
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

        if not (exp_edge["src"].loc[0], exp_edge["dst"].loc[0]) == (src, dst):
            print(
                "[ERR] Invalid edge: " "There is no edge src {} dst {}".format(src, dst)
            )
            invalid_edge += 1

    assert invalid_edge == 0
    assert invalid_seeds == 0


@pytest.mark.parametrize("graph_file", DATASETS_SMALL)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize("max_depth", [None])
def test_random_walks_invalid_max_dept(graph_file, directed, max_depth):
    with pytest.raises(TypeError):
        df, offsets, seeds = calc_random_walks(
            graph_file, directed=directed, max_depth=max_depth
        )


@pytest.mark.cugraph_ops
@pytest.mark.parametrize("graph_file", DATASETS_SMALL)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_random_walks_coalesced(graph_file, directed):
    max_depth = random.randint(2, 10)
    df_G = graph_file.get_edgelist()
    path_data, seeds = calc_random_walks(graph_file, directed, max_depth=max_depth)
    check_random_walks(path_data, seeds, df_G)

    # Check path query output
    df = cugraph.rw_path(len(seeds), path_data[2])
    v_offsets = [0] + path_data[2].cumsum()[:-1].to_numpy().tolist()
    w_offsets = [0] + (path_data[2] - 1).cumsum()[:-1].to_numpy().tolist()

    assert_series_equal(df["weight_sizes"], path_data[2] - 1, check_names=False)
    assert df["vertex_offsets"].to_numpy().tolist() == v_offsets
    assert df["weight_offsets"].to_numpy().tolist() == w_offsets


@pytest.mark.cugraph_ops
@pytest.mark.parametrize("graph_file", DATASETS_SMALL)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_random_walks_padded(graph_file, directed):
    max_depth = random.randint(2, 10)
    path_data, seeds = calc_random_walks(
        graph_file, directed, max_depth=max_depth, use_padding=True
    )
    v_paths = path_data[0]
    e_weights = path_data[1]
    assert len(v_paths) == max_depth * len(seeds)
    assert len(e_weights) == (max_depth - 1) * len(seeds)


"""@pytest.mark.parametrize("graph_file", utils.DATASETS_SMALL)
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
