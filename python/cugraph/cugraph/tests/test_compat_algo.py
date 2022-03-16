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
import cugraph.compat.nx as nx

def test_connectivity():
    expected = list([{1, 2, 3, 4, 5}, {8, 9, 7}])
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3,4), (4,5)])
    G.add_edges_from([(7,8),(8,9),(7,9)])
    print(list(nx.connected_components(G)))
    assert list(nx.connected_components(G)) == expected

def test_pagerank():
    G = nx.DiGraph()
    [G.add_node(k) for k in ["A", "B", "C", "D", "E", "F", "G"]]
    G.add_edges_from([('G','A'), ('A','G'),('B','A'),
                    ('C','A'),('A','C'),('A','D'),
                    ('E','A'),('F','A'),('D','B'),
                    ('D','F')])
    ppr1 = nx.pagerank(G)
    assert type(ppr1) == dict
    print("reached compat pagerank")
    print(f'Page rank value: {ppr1}')
    pos = nx.spiral_layout(G)
    print (ppr1)




