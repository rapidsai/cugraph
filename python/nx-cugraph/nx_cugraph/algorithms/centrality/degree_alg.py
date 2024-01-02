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
from nx_cugraph.convert import _to_directed_graph, _to_graph
from nx_cugraph.utils import networkx_algorithm, not_implemented_for

__all__ = ["degree_centrality", "in_degree_centrality", "out_degree_centrality"]


@networkx_algorithm(version_added="23.12")
def degree_centrality(G):
    G = _to_graph(G)
    if len(G) <= 1:
        return dict.fromkeys(G, 1)
    deg = G._degrees_array()
    centrality = deg * (1 / (len(G) - 1))
    return G._nodearray_to_dict(centrality)


@not_implemented_for("undirected")
@networkx_algorithm(version_added="23.12")
def in_degree_centrality(G):
    G = _to_directed_graph(G)
    if len(G) <= 1:
        return dict.fromkeys(G, 1)
    deg = G._in_degrees_array()
    centrality = deg * (1 / (len(G) - 1))
    return G._nodearray_to_dict(centrality)


@not_implemented_for("undirected")
@networkx_algorithm(version_added="23.12")
def out_degree_centrality(G):
    G = _to_directed_graph(G)
    if len(G) <= 1:
        return dict.fromkeys(G, 1)
    deg = G._out_degrees_array()
    centrality = deg * (1 / (len(G) - 1))
    return G._nodearray_to_dict(centrality)
