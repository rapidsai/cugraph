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
from packaging.version import parse

nxver = parse(nx.__version__)

if nxver.major == 3 and nxver.minor < 2:
    pytest.skip("Need NetworkX >=3.2 to test clustering", allow_module_level=True)


def test_selfloops():
    G = nx.complete_graph(5)
    H = nx.complete_graph(5)
    H.add_edge(0, 0)
    H.add_edge(1, 1)
    H.add_edge(2, 2)
    # triangles
    expected = nx.triangles(G)
    assert expected == nx.triangles(H)
    assert expected == nx.triangles(G, backend="cugraph")
    assert expected == nx.triangles(H, backend="cugraph")
    # average_clustering
    expected = nx.average_clustering(G)
    assert expected == nx.average_clustering(H)
    assert expected == nx.average_clustering(G, backend="cugraph")
    assert expected == nx.average_clustering(H, backend="cugraph")
    # clustering
    expected = nx.clustering(G)
    assert expected == nx.clustering(H)
    assert expected == nx.clustering(G, backend="cugraph")
    assert expected == nx.clustering(H, backend="cugraph")
    # transitivity
    expected = nx.transitivity(G)
    assert expected == nx.transitivity(H)
    assert expected == nx.transitivity(G, backend="cugraph")
    assert expected == nx.transitivity(H, backend="cugraph")
