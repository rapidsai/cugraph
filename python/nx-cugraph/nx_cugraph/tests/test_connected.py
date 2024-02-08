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

import nx_cugraph as nxcg


def test_connected_isolated_nodes():
    G = nx.complete_graph(4)
    G.add_node(max(G) + 1)
    assert nx.is_connected(G) is False
    assert nxcg.is_connected(G) is False
    assert nx.number_connected_components(G) == 2
    assert nxcg.number_connected_components(G) == 2
    assert sorted(nx.connected_components(G)) == [{0, 1, 2, 3}, {4}]
    assert sorted(nxcg.connected_components(G)) == [{0, 1, 2, 3}, {4}]
    assert nx.node_connected_component(G, 0) == {0, 1, 2, 3}
    assert nxcg.node_connected_component(G, 0) == {0, 1, 2, 3}
    assert nx.node_connected_component(G, 4) == {4}
    assert nxcg.node_connected_component(G, 4) == {4}
