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

import cugraph.experimental.compat.nx as nx


def test_connectivity():
    # Tests a run of a native nx algorithm that hasnt been overridden.
    expected = [{1, 2, 3, 4, 5}, {8, 9, 7}]
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
    G.add_edges_from([(7, 8), (8, 9), (7, 9)])
    assert list(nx.connected_components(G)) == expected


def test_pagerank_result_type():
    G = nx.DiGraph()
    [G.add_node(k) for k in ["A", "B", "C", "D", "E", "F", "G"]]
    G.add_edges_from(
        [
            ("G", "A"),
            ("A", "G"),
            ("B", "A"),
            ("C", "A"),
            ("A", "C"),
            ("A", "D"),
            ("E", "A"),
            ("F", "A"),
            ("D", "B"),
            ("D", "F"),
        ]
    )
    ppr1 = nx.pagerank(G)
    # This just tests that the right type is returned.
    assert isinstance(ppr1, dict)
