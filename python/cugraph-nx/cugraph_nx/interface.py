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
from __future__ import annotations

from typing import TYPE_CHECKING

import cugraph_nx as cnx

from . import algorithms

if TYPE_CHECKING:
    import networkx as nx


class Dispatcher:
    is_strongly_connected = algorithms.is_strongly_connected

    @staticmethod
    def convert_from_nx(graph: nx.Graph, weight=None, *, name=None) -> cnx.Graph:
        return cnx.from_networkx(graph, edge_attr=weight)

    @staticmethod
    def convert_to_nx(obj, *, name=None):
        if isinstance(obj, cnx.Graph):
            return cnx.to_networkx(obj)
        return obj

    @staticmethod
    def on_start_tests(items):
        pass
