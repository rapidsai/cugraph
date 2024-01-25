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


def test_generic_bfs_edges():
    # generic_bfs_edges currently isn't exercised by networkx tests
    Gnx = nx.karate_club_graph()
    Gcg = nx.karate_club_graph(backend="cugraph")
    for depth_limit in (0, 1, 2):
        for source in Gnx:
            # Some ordering is arbitrary, so I think there's a chance
            # this test may fail if networkx or nx-cugraph changes.
            nx_result = nx.generic_bfs_edges(Gnx, source, depth_limit=depth_limit)
            cg_result = nx.generic_bfs_edges(Gcg, source, depth_limit=depth_limit)
            assert sorted(nx_result) == sorted(cg_result), (source, depth_limit)
