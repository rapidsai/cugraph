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
import pytest

import nx_cugraph as nxcg

from .testing_utils import assert_graphs_equal


def _create_Gs():
    rv = []
    rv.append(nx.DiGraph())
    G = nx.DiGraph()
    G.add_edge(0, 1)
    G.add_edge(1, 0)
    rv.append(G)
    G = G.copy()
    G.add_edge(0, 2)
    rv.append(G)
    G = G.copy()
    G.add_edge(1, 1)
    rv.append(G)
    G = nx.DiGraph()
    G.add_edge(0, 1, x=1, y=2)
    G.add_edge(1, 0, x=10, z=3)
    rv.append(G)
    G = G.copy()
    G.add_edge(0, 2, a=42)
    rv.append(G)
    G = G.copy()
    G.add_edge(1, 1, a=4)
    rv.append(G)
    return rv


@pytest.mark.parametrize("Gnx", _create_Gs())
@pytest.mark.parametrize("reciprocal", [False, True])
def test_to_undirected_directed(Gnx, reciprocal):
    Gcg = nxcg.CudaDiGraph(Gnx)
    assert_graphs_equal(Gnx, Gcg)
    Hnx1 = Gnx.to_undirected(reciprocal=reciprocal)
    Hcg1 = Gcg.to_undirected(reciprocal=reciprocal)
    assert_graphs_equal(Hnx1, Hcg1)
    Hnx2 = Hnx1.to_directed()
    Hcg2 = Hcg1.to_directed()
    assert_graphs_equal(Hnx2, Hcg2)


def test_multidigraph_to_undirected():
    Gnx = nx.MultiDiGraph()
    Gnx.add_edge(0, 1)
    Gnx.add_edge(0, 1)
    Gnx.add_edge(1, 0)
    Gcg = nxcg.CudaMultiDiGraph(Gnx)
    with pytest.raises(NotImplementedError):
        Gcg.to_undirected()
