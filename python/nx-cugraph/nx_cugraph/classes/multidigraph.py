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

from .digraph import DiGraph
from .multigraph import MultiGraph

__all__ = ["MultiDiGraph"]

networkx_api = nxcg.utils.decorators.networkx_class(nx.MultiDiGraph)


class MultiDiGraph(MultiGraph, DiGraph):
    @classmethod
    @networkx_api
    def is_directed(cls) -> bool:
        return True

    @classmethod
    def to_networkx_class(cls) -> type[nx.MultiDiGraph]:
        return nx.MultiDiGraph

    ##########################
    # NetworkX graph methods #
    ##########################

    @networkx_api
    def to_undirected(self, reciprocal=False, as_view=False):
        raise NotImplementedError
