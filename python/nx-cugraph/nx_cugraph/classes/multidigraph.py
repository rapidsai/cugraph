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
gpu_cpu_api = nxcg.utils.decorators._gpu_cpu_api(nx.MultiDiGraph, __name__)


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
    __contains__ = gpu_cpu_api("__contains__")
    __len__ = gpu_cpu_api("__len__")
    __iter__ = gpu_cpu_api("__iter__")
    clear = DiGraph.clear
    clear_edges = DiGraph.clear_edges
    get_edge_data = gpu_cpu_api("get_edge_data", edge_data=True)
    has_edge = gpu_cpu_api("has_edge")
    neighbors = gpu_cpu_api("neighbors")
    has_node = gpu_cpu_api("has_node")
    nbunch_iter = gpu_cpu_api("nbunch_iter")
    number_of_edges = MultiGraph.number_of_edges
    number_of_nodes = gpu_cpu_api("number_of_nodes")
    order = gpu_cpu_api("order")
    successors = gpu_cpu_api("successors")


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
