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


# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import pytest
from cugraph.tests import utils

import networkx as nx


def test_nx_gbuilder():

    # Create an empty graph
    G = nx.Graph()
    assert G.number_of_edges() == 0
    assert G.number_of_nodes() == 0

    # Add a node
    G.add_node(1)
    assert G.number_of_edges() == 0
    assert G.number_of_nodes() == 1

    # Add some edges
    G.add_edges_from([(1, 2), (1, 3)])
    assert G.number_of_edges() == 2
    assert G.number_of_nodes() == 3

    # Add some duplicates
    G.add_edges_from([(1, 2), (1, 3)])
    G.add_node(1)
    G.add_edge(1, 2)
    assert G.number_of_edges() == 2
    assert G.number_of_nodes() == 3

    # Add nodes with a property from a list
    G.add_nodes_from([(4, {"color": "red"}), (5, {"color": "green"}), ])
    assert G.nodes[4]["color"] == "red"

    G.add_node("spam")        # adds node "spam"
    G.add_nodes_from("spam")  # adds 4 nodes: 's', 'p', 'a', 'm'
    G.add_edge(3, 'm')
    assert G.number_of_edges() == 3
    assert G.number_of_nodes() == 10
    assert list(G.nodes) == [1, 2, 3, 4, 5, 'spam', 's', 'p', 'a', 'm']
    # remove nodes
    G.remove_node(2)
    G.remove_nodes_from("spam")
    assert list(G.nodes) == [1, 3, 4, 5, 'spam']
    G.remove_edge(1, 3)

    # Access edge attributes
    G = nx.Graph([(1, 2, {"color": "yellow"})])
    assert G[1][2] == {'color': 'yellow'}
    assert G.edges[1, 2] == {'color': 'yellow'}


def test_nx_graph_functions():
    # test adjacency
    FG = nx.Graph()
    FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75),
                                (2, 4, 1.2), (3, 4, 0.375)])
    for n, nbrs in FG.adj.items():
        for nbr, eattr in nbrs.items():
            wt = eattr['weight']
            if wt < 0.5:
                assert FG[n][nbr]['weight'] < 0.5
    # accessing graph edges
    for (u, v, wt) in FG.edges.data('weight'):
        if wt < 0.5:
            assert FG[u][v]['weight'] <= 0.5
        else:
            assert FG[u][v]['weight'] > 0.5


def test_nx_analysis():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3)])
    G.add_node("spam")       # adds node "spam"
    assert list(nx.connected_components(G)) == [{1, 2, 3}, {'spam'}]
    assert sorted(d for n, d in G.degree()) == [0, 1, 1, 2]
    assert nx.clustering(G) == {1: 0, 2: 0, 3: 0, 'spam': 0}
    assert list(nx.bfs_edges(G, 1)) == [(1, 2), (1, 3)]


@pytest.mark.parametrize(
    "graph_file",
    [utils.RAPIDS_DATASET_ROOT_DIR_PATH/"dolphins.csv"])
def test_with_dolphins(graph_file):

    df = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)
    G = nx.from_pandas_edgelist(df, create_using=nx.Graph(),
                                source="0", target="1", edge_attr="weight")

    assert G.degree(0) == 6
    assert G.degree(14) == 12
    assert G.degree(15) == 7
    assert G.degree(40) == 8
    assert G.degree(42) == 6
    assert G.degree(47) == 6
    assert G.degree(17) == 9
