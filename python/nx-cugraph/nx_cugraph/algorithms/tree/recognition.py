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
import cupy as cp
import networkx as nx

import nx_cugraph as nxcg
from nx_cugraph.convert import _to_directed_graph, _to_graph
from nx_cugraph.utils import networkx_algorithm, not_implemented_for

__all__ = ["is_arborescence", "is_branching", "is_forest", "is_tree"]


@not_implemented_for("undirected")
@networkx_algorithm(plc="weakly_connected_components", version_added="24.02")
def is_arborescence(G):
    G = _to_directed_graph(G)
    return is_tree(G) and int(G._in_degrees_array().max()) <= 1


@not_implemented_for("undirected")
@networkx_algorithm(plc="weakly_connected_components", version_added="24.02")
def is_branching(G):
    G = _to_directed_graph(G)
    return is_forest(G) and int(G._in_degrees_array().max()) <= 1


@networkx_algorithm(plc="weakly_connected_components", version_added="24.02")
def is_forest(G):
    G = _to_graph(G)
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("G has no nodes.")
    if is_directed := G.is_directed():
        connected_components = nxcg.weakly_connected_components
    else:
        connected_components = nxcg.connected_components
    for components in connected_components(G):
        node_ids = G._list_to_nodearray(list(components))
        # TODO: create utilities for creating subgraphs
        mask = cp.isin(G.src_indices, node_ids) & cp.isin(G.dst_indices, node_ids)
        # A tree must have an edge count equal to the number of nodes minus the
        # tree's root node.
        if is_directed:
            if int(cp.count_nonzero(mask)) != len(components) - 1:
                return False
        else:
            src_indices = G.src_indices[mask]
            dst_indices = G.dst_indices[mask]
            if int(cp.count_nonzero(src_indices <= dst_indices)) != len(components) - 1:
                return False
    return True


@networkx_algorithm(plc="weakly_connected_components", version_added="24.02")
def is_tree(G):
    G = _to_graph(G)
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("G has no nodes.")
    if G.is_directed():
        is_connected = nxcg.is_weakly_connected
    else:
        is_connected = nxcg.is_connected
    # A tree must have an edge count equal to the number of nodes minus the
    # tree's root node.
    return len(G) - 1 == G.number_of_edges() and is_connected(G)
