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
from nx_cugraph.convert import _to_directed_graph
from nx_cugraph.utils import networkx_algorithm, not_implemented_for

from .connected import (
    _connected_components,
    _is_connected,
    _number_connected_components,
)

__all__ = [
    "number_weakly_connected_components",
    "weakly_connected_components",
    "is_weakly_connected",
]


@not_implemented_for("undirected")
@networkx_algorithm(plc="weakly_connected_components", version_added="24.02")
def weakly_connected_components(G):
    G = _to_directed_graph(G)
    return _connected_components(G, symmetrize="union")


@not_implemented_for("undirected")
@networkx_algorithm(plc="weakly_connected_components", version_added="24.02")
def number_weakly_connected_components(G):
    G = _to_directed_graph(G)
    return _number_connected_components(G, symmetrize="union")


@not_implemented_for("undirected")
@networkx_algorithm(plc="weakly_connected_components", version_added="24.02")
def is_weakly_connected(G):
    G = _to_directed_graph(G)
    return _is_connected(G, symmetrize="union")
