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
# import random

import pytest
# from cudf.testing import assert_series_equal

from cugraph.tests import utils
# import cugraph


# =============================================================================
# Parameters
# =============================================================================
DIRECTED_GRAPH_OPTIONS = [False, True]
WEIGHTED_GRAPH_OPTIONS = [False, True]
DATASETS = [pytest.param(d) for d in utils.DATASETS]
DATASETS_SMALL = [pytest.param(d) for d in utils.DATASETS_SMALL]


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def calc_node2vec(graph_file,
                  directed=False,
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

    p : double

    q : double
    """
    G = utils.generate_cugraph_graph_from_file(
        graph_file, directed=directed, edgevals=True)
    assert G is not None

    k = random.randint(1, 10)
    start_vertices = random.sample(range(G.number_of_vertices()), k)
    vertex_paths, edge_weights, vertex_path_sizes = cugraph.node2vec(
        G, start_vertices, max_depth, use_padding, p, q)

    return (vertex_paths, edge_weights, vertex_path_sizes), start_vertices


def check_node2vec(path_data, seeds, df_G=None):
    invalid_edge = 0
    invalid_seeds = 0
    offsets_idx = 0
    next_path_idx = 0
    v_paths = path_data[0]
    sizes = path_data[2].to_numpy().tolist()

    for s in sizes:
        for i in range(next_path_idx, next_path_idx+s-1):
            src, dst = v_paths.iloc[i],  v_paths.iloc[i+1]
            if i == next_path_idx and src != seeds[offsets_idx]:
                invalid_seeds += 1
                print(
                        "[ERR] Invalid seed: "
                        " src {} != src {}"
                        .format(src, seeds[offsets_idx])
                    )
        offsets_idx += 1
        next_path_idx += s

        exp_edge = df_G.loc[
            (df_G['src'] == (src)) & (
                df_G['dst'] == (dst))].reset_index(drop=True)

        if not (exp_edge['src'].loc[0], exp_edge['dst'].loc[0]) == (src, dst):
            print(
                    "[ERR] Invalid edge: "
                    "There is no edge src {} dst {}"
                    .format(src, dst)
                )
            invalid_edge += 1

    assert invalid_edge == 0
    assert invalid_seeds == 0


@pytest.mark.parametrize("graph_file", utils.DATASETS_SMALL)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize("max_depth", [None. -1])
def test_node2vec_invalid_max_depth(graph_file,
                                    directed,
                                    max_depth):
    with pytest.raises(TypeError):
        df, offsets, seeds = calc_node2vec(
            graph_file,
            directed=directed,
            max_depth=max_depth,
            use_padding=use_padding,
            p=p,
            q=q
        )


@pytest.mark.parametrize("graph_file", utils.DATASETS_SMALL)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_node2vec_coalesced():
    max_depth = random.randint(2, 10)
    df_G = utils.read_csv_file(graph_file)
    df_G.rename(
        columns={"0": "src", "1": "dst", "2": "weight"}, inplace=True)
    path_data, seeds = calc_node2vec(
        graph_file,
        directed,
        max_depth=max_depth,
        use_padding=False,
        p,
        q
    )
    check_random_walks(path_data, seeds, df_G)

    # Check path query output
    # df = cugraph.rw_path(len(seeds), path_data[2])
    # v_offsets = [0] + path_data[2].cumsum()[:-1].to_numpy().tolist()
    # w_offsets = [0] + (path_data[2]-1).cumsum()[:-1].to_numpy().tolist()

    # assert_series_equal(df['weight_sizes'], path_data[2]-1,
    #                     check_names=False)
    # assert df['vertex_offsets'].to_numpy().tolist() == v_offsets
    # assert df['weight_offsets'].to_numpy().tolist() == w_offsets


@pytest.mark.parametrize("graph_file", utils.DATASETS_SMALL)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_node2vec_padded(
    graph_file,
    directed,
    p,
    q
):
    max_depth = random.randint(2, 10)
    df_G = utils.read_csv_file(graph_file)
    df_G.rename(
        columns={"0": "src", "1": "dst", "2": "weight"}, inplace=True)
    path_data, seeds = calc_node2vec(
        graph_file,
        directed,
        max_depth=max_depth,
        use_padding=True,
        p,
        q
    )
    v_paths = path_data[0]
    e_weights = path_data[1]
    assert len(v_paths) == max_depth*len(seeds)
    assert len(e_weights) == (max_depth - 1)*len(seeds)


@pytest.mark.parametrize("graph_file", utils.DATASETS_SMALL)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_node2vec_nx(graph_file, directed):
    max_depth = random.randint(2, 10)
    nx_G = utils.create_obj_from_csv(graph_file, nx.Graph, directed=directed)
    nx_G.rename(
        columns={"0": "src", "1": "dst", "2": "weight"}, inplace=True)
    k = random.randint(1, 10)
    start_vertices = random.sample(range(G.number_of_vertices()), k)

    vertex_paths, edge_weights, vertex_path_sizes = cugraph.node2vec(
            G, start_vertices, max_depth, True, p, q)

    assert len(vertex_paths) == max_depth * len(start_vertices)
    assert len(edge_weights) == (max_depth - 1) * len(start_vertices)
