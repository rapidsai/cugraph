# Copyright (c) 2020-2021, NVIDIA CORPORATION.:
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

import cudf
from cugraph.tests import utils
import cugraph
import random


# =============================================================================
# Parameters
# =============================================================================
DIRECTED_GRAPH_OPTIONS = [False, True]
WEIGHTED_GRAPH_OPTIONS = [False, True]
DATASETS = [pytest.param(d) for d in utils.DATASETS]
DATASETS_SMALL = [pytest.param(d) for d in utils.DATASETS_SMALL]


def calc_random_walks(
    graph_file,
    directed=False,
    max_depth=None
):
    """
    compute random walks for each nodes in 'start_vertices'

    parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
        The graph can be either directed (DiGraph) or undirected (Graph).
        Weights in the graph are ignored.
        Use weight parameter if weights need to be considered
        (currently not supported)

    start_vertices : int or list or cudf.Series
        A single node or a list or a cudf.Series of nodes from which to run
        the random walks

    max_depth : int
        The maximum depth of the random walks


    Returns
    -------
    random_walks_edge_lists : cudf.DataFrame
        GPU data frame containing all random walks sources identifiers,
        destination identifiers, edge weights

    seeds_offsets: cudf.Series
        Series containing the starting offset in the returned edge list
        for each vertex in start_vertices.
    """
    G = utils.generate_cugraph_graph_from_file(
        graph_file, directed=directed, edgevals=True)
    assert G is not None

    k = random.randint(1, 10)
    start_vertices = random.sample(range(G.number_of_vertices()), k)
    df, offsets = cugraph.random_walks(G, start_vertices, max_depth)

    return df, offsets, start_vertices


def check_random_walks(df, offsets, seeds, df_G=None):
    invalid_edge = 0
    invalid_seeds = 0
    invalid_weight = 0
    offsets_idx = 0
    for i in range(len(df.index)):
        src, dst, weight = df.iloc[i].to_array()
        if i == offsets[offsets_idx]:
            if df['src'].iloc[i] != seeds[offsets_idx]:
                invalid_seeds += 1
                print(
                        "[ERR] Invalid seed: "
                        " src {} != src {}"
                        .format(df['src'].iloc[i], offsets[offsets_idx])
                    )
            offsets_idx += 1

        edge = df.loc[(df['src'] == (src)) & (df['dst'] == (dst))].reset_index(
            drop=True)
        exp_edge = df_G.loc[
            (df_G['src'] == (src)) & (
                df_G['dst'] == (dst))].reset_index(drop=True)

        if not exp_edge.equals(edge[:1]):
            print(
                    "[ERR] Invalid edge: "
                    "There is no edge src {} dst {} weight {}"
                    .format(src, dst, weight)
                )
            invalid_weight += 1

    assert invalid_edge == 0
    assert invalid_seeds == 0
    assert invalid_weight == 0

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def prepare_test():
    gc.collect()


@pytest.mark.parametrize("graph_file", utils.DATASETS_SMALL)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize("max_depth", [None])
def test_random_walks_invalid_max_dept(
    graph_file,
    directed,
    max_depth
):
    prepare_test()
    with pytest.raises(TypeError):
        df, offsets, seeds = calc_random_walks(
            graph_file,
            directed=directed,
            max_depth=max_depth
        )


@pytest.mark.parametrize("graph_file", utils.DATASETS_SMALL)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_random_walks(
    graph_file,
    directed
):
    max_depth = random.randint(2, 10)
    df_G = utils.read_csv_file(graph_file)
    df_G.rename(
        columns={"0": "src", "1": "dst", "2": "weight"}, inplace=True)
    df, offsets, seeds = calc_random_walks(
        graph_file,
        directed,
        max_depth=max_depth
    )
    check_random_walks(df, offsets, seeds, df_G)


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
        G = cugraph.DiGraph()
    else:
        G = cugraph.Graph()
    G.from_cudf_edgelist(df_G, source=['src', 'src_0'],
                         destination=['dst', 'dst_0'],
                         edge_attr="weight")

    k = random.randint(1, 10)
    start_vertices = random.sample(G.nodes().to_array().tolist(), k)

    seeds = cudf.DataFrame()
    seeds['v'] = start_vertices
    seeds['v_0'] = seeds['v'] + 1000

    df, offsets = cugraph.random_walks(G, seeds, max_depth)

    check_random_walks(df, offsets, seeds, df_G)
"""
