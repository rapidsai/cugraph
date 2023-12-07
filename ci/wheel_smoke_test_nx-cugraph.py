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

import math

import networkx as nx
import nx_cugraph as nxcg


if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])

    nx_result = nx.betweenness_centrality(G)
    # nx_cugraph is intended to be called via the NetworkX dispatcher, like
    # this:
    #    nxcu_result = nx.betweenness_centrality(G, backend="cugraph")
    #
    # but here it is being called directly since the NetworkX version that
    # supports the "backend" kwarg may not be available in the testing env.
    nxcu_result = nxcg.betweenness_centrality(G)

    nx_nodes, nxcu_nodes = nx_result.keys(), nxcu_result.keys()
    assert nxcu_nodes == nx_nodes
    for node_id in nx_nodes:
        nx_bc, nxcu_bc = nx_result[node_id], nxcu_result[node_id]
        assert math.isclose(nx_bc, nxcu_bc, rel_tol=1e-6), \
            f"bc for {node_id=} exceeds tolerance: {nx_bc=}, {nxcu_bc=}"
