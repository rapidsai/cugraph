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
import cupy as cp
import networkx as nx

from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import networkx_algorithm

__all__ = [
    "is_negatively_weighted",
    "number_of_selfloops",
]


@networkx_algorithm(version_added="24.04")
def is_negatively_weighted(G, edge=None, weight="weight"):
    G = _to_graph(G, weight)
    if edge is not None:
        data = G.get_edge_data(*edge)
        if data is None:
            raise nx.NetworkXError(f"Edge {edge!r} does not exist.")
        return weight in data and data[weight] < 0
    if weight not in G.edge_values:
        return False
    edge_vals = G.edge_values[weight]
    if weight in G.edge_masks:
        edge_vals = edge_vals[G.edge_masks[weight]]
    return bool((edge_vals < 0).any())


@networkx_algorithm(version_added="23.12")
def number_of_selfloops(G):
    G = _to_graph(G)
    is_selfloop = G.src_indices == G.dst_indices
    return int(cp.count_nonzero(is_selfloop))


@number_of_selfloops._should_run
def _(G):
    return "Fast algorithm; not worth converting."
