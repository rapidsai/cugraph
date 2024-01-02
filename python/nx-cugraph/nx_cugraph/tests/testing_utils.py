# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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


def assert_graphs_equal(Gnx, Gcg):
    assert isinstance(Gnx, nx.Graph)
    assert isinstance(Gcg, nxcg.Graph)
    assert Gnx.number_of_nodes() == Gcg.number_of_nodes()
    assert Gnx.number_of_edges() == Gcg.number_of_edges()
    assert Gnx.is_directed() == Gcg.is_directed()
    assert Gnx.is_multigraph() == Gcg.is_multigraph()
    G = nxcg.to_networkx(Gcg)
    rv = nx.utils.graphs_equal(G, Gnx)
    if not rv:
        print("GRAPHS ARE NOT EQUAL!")
        assert sorted(G) == sorted(Gnx)
        assert sorted(G._adj) == sorted(Gnx._adj)
        assert sorted(G._node) == sorted(Gnx._node)
        for k in sorted(G._adj):
            print(k, sorted(G._adj[k]), sorted(Gnx._adj[k]))
        print(nx.to_scipy_sparse_array(G).todense())
        print(nx.to_scipy_sparse_array(Gnx).todense())
        print(G.graph)
        print(Gnx.graph)
    assert rv
