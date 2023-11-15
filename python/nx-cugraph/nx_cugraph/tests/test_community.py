# Copyright (c) 2023, NVIDIA CORPORATION.
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


def test_louvain_isolated_nodes():
    is_nx_30_or_31 = hasattr(nx.classes, "backends")

    def check(left, right):
        assert len(left) == len(right)
        assert set(map(frozenset, left)) == set(map(frozenset, right))

    # Empty graph (no nodes)
    G = nx.Graph()
    if is_nx_30_or_31:
        with pytest.raises(ZeroDivisionError):
            nx.community.louvain_communities(G)
    else:
        nx_result = nx.community.louvain_communities(G)
        cg_result = nxcg.community.louvain_communities(G)
        check(nx_result, cg_result)
    # Graph with no edges
    G.add_nodes_from(range(5))
    if is_nx_30_or_31:
        with pytest.raises(ZeroDivisionError):
            nx.community.louvain_communities(G)
    else:
        nx_result = nx.community.louvain_communities(G)
        cg_result = nxcg.community.louvain_communities(G)
        check(nx_result, cg_result)
    # Graph with isolated nodes
    G.add_edge(1, 2)
    nx_result = nx.community.louvain_communities(G)
    cg_result = nxcg.community.louvain_communities(G)
    check(nx_result, cg_result)
    # Another one
    G.add_edge(4, 4)
    nx_result = nx.community.louvain_communities(G)
    cg_result = nxcg.community.louvain_communities(G)
    check(nx_result, cg_result)
