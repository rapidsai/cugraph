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


@pytest.mark.parametrize("graph_file", utils.DATASETS_SMALL)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_random_walks_coalesced(
    graph_file,
    directed
):
    df, seeds = calc_node2vec(
        graph_file,
        directed=directed,
        max_depth=3,
        use_padding=False,
        p=0.8,
        q=0.5
    )
    # Check that weights match up with paths


@pytest.mark.parametrize("graph_file", utils.DATASETS_SMALL)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_random_walks_padded(
    graph_file,
    directed
):
    df, seeds = calc_node2vec(
        graph_file,
        directed=directed,
        max_depth=3,
        use_padding=True,
        p=0.8,
        q=0.5
    )
    # Check that weights match up with paths

    # Check that path sizes matches up correctly with paths


@pytest.mark.parametrize("graph_file", utils.DATASETS_SMALL)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize("max_depth", [None, -1])
@pytest.mark.parametrize("p", [None, -1])
def test_random_walks_invalid(
    graph_file,
    directed,
    max_depth,
    p
):
    # Tests for invalid max depth, p, and q
    use_padding = True
    q = 1.0
    with pytest.raises(TypeError):
        df, seeds = calc_node2vec(
            graph_file,
            directed=directed,
            max_depth=max_depth,
            use_padding=use_padding,
            p=p,
            q=q
        )


@pytest.mark.parametrize("graph_file", utils.DATASETS_SMALL)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_random_walks_nx(
    graph_file,
    directed
):
    df, seeds = calc_node2vec(
        graph_file,
        directed=directed,
        max_depth=3,
        use_padding=True,
        p=0.8,
        q=0.5
    )
