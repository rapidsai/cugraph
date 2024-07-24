# Copyright (c) 2024, NVIDIA CORPORATION.
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
import networkx as nx
import pytest

import nx_cugraph as nxcg

from .testing_utils import assert_graphs_equal


@pytest.mark.parametrize(
    "create_using", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_relabel(create_using):
    G = nx.complete_graph(3, create_using=create_using)
    Hnx = nx.relabel_nodes(G, {2: 1})
    Hcg = nxcg.relabel_nodes(G, {2: 1})
    assert_graphs_equal(Hnx, Hcg)

    G.add_edge(0, 2, a=11)
    G.add_edge(1, 2, b=22)
    Hnx = nx.relabel_nodes(G, {2: 10, 1: 10})
    Hcg = nxcg.relabel_nodes(G, {2: 10, 1: 10})
    assert_graphs_equal(Hnx, Hcg)

    G = nx.path_graph(3, create_using=create_using)
    Hnx = nx.relabel_nodes(G, {2: 0})
    Hcg = nxcg.relabel_nodes(G, {2: 0})
    assert_graphs_equal(Hnx, Hcg)


@pytest.mark.parametrize("create_using", [nx.MultiGraph, nx.MultiDiGraph])
def test_relabel_multigraph(create_using):
    G = nx.empty_graph(create_using=create_using)
    G.add_edge(0, 1, "x", a=11)
    G.add_edge(0, 2, "y", a=10, b=6)
    G.add_edge(0, 0, c=7)
    G.add_edge(0, 0, "x", a=-1, b=-1, c=-1)
    Hnx = nx.relabel_nodes(G, {0: 1, 2: 1})
    Hcg = nxcg.relabel_nodes(G, {0: 1, 2: 1})
    assert_graphs_equal(Hnx, Hcg)
    Hnx = nx.relabel_nodes(G, {2: 3, 1: 3, 0: 3})
    Hcg = nxcg.relabel_nodes(G, {2: 3, 1: 3, 0: 3})
    assert_graphs_equal(Hnx, Hcg)


def test_relabel_nx_input():
    G = nx.complete_graph(3)
    with pytest.raises(RuntimeError, match="Using `copy=False` is invalid"):
        nxcg.relabel_nodes(G, {0: 1}, copy=False)
    Hnx = nx.relabel_nodes(G, {0: 1}, copy=True)
    Hcg = nxcg.relabel_nodes(G, {0: 1}, copy=True)
    assert_graphs_equal(Hnx, Hcg)
