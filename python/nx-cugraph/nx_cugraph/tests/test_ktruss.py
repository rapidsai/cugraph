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


@pytest.mark.parametrize(
    "get_graph", [nx.florentine_families_graph, nx.les_miserables_graph]
)
def test_k_truss(get_graph):
    Gnx = get_graph()
    Gcg = nxcg.from_networkx(Gnx, preserve_all_attrs=True)
    for k in range(10):
        Hnx = nx.k_truss(Gnx, k)
        Hcg = nxcg.k_truss(Gcg, k)
        assert nx.utils.graphs_equal(Hnx, nxcg.to_networkx(Hcg))
        if Hnx.number_of_edges() == 0:
            break
