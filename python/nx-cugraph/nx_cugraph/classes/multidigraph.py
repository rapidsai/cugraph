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
from __future__ import annotations

import networkx as nx

import nx_cugraph as nxcg

from .digraph import CudaDiGraph, DiGraph
from .graph import Graph
from .multigraph import CudaMultiGraph, MultiGraph

__all__ = ["CudaMultiDiGraph", "MultiDiGraph"]

networkx_api = nxcg.utils.decorators.networkx_class(nx.MultiDiGraph)


class MultiDiGraph(nx.MultiDiGraph, MultiGraph, DiGraph):
    name = Graph.name
    _node = Graph._node
    _adj = DiGraph._adj
    _succ = DiGraph._succ
    _pred = DiGraph._pred

    @classmethod
    @networkx_api
    def is_directed(cls) -> bool:
        return True

    @classmethod
    @networkx_api
    def is_multigraph(cls) -> bool:
        return True

    @classmethod
    def to_cudagraph_class(cls) -> type[CudaMultiDiGraph]:
        return CudaMultiDiGraph

    @classmethod
    def to_networkx_class(cls) -> type[nx.MultiDiGraph]:
        return nx.MultiDiGraph

    ##########################
    # Networkx graph methods #
    ##########################

    # Dispatch to nx.MultiDiGraph or CudaMultiDiGraph
    __contains__ = Graph.__dict__["__contains__"]
    __len__ = Graph.__dict__["__len__"]
    __iter__ = Graph.__dict__["__iter__"]
    get_edge_data = Graph.__dict__["get_edge_data"]
    has_edge = Graph.__dict__["has_edge"]
    neighbors = Graph.__dict__["neighbors"]
    has_node = Graph.__dict__["has_node"]
    nbunch_iter = Graph.__dict__["nbunch_iter"]
    number_of_nodes = Graph.__dict__["number_of_nodes"]
    order = Graph.__dict__["order"]
    successors = Graph.__dict__["neighbors"]  # Alias

    clear = Graph.clear
    clear_edges = Graph.clear_edges
    number_of_edges = Graph.number_of_edges


class CudaMultiDiGraph(CudaMultiGraph, CudaDiGraph):
    is_directed = classmethod(MultiDiGraph.is_directed.__func__)
    is_multigraph = classmethod(MultiDiGraph.is_multigraph.__func__)
    to_cudagraph_class = classmethod(MultiDiGraph.to_cudagraph_class.__func__)
    to_networkx_class = classmethod(MultiDiGraph.to_networkx_class.__func__)

    @classmethod
    def _to_compat_graph_class(cls) -> type[MultiDiGraph]:
        return MultiDiGraph

    ##########################
    # NetworkX graph methods #
    ##########################

    @networkx_api
    def to_undirected(self, reciprocal=False, as_view=False):
        raise NotImplementedError
