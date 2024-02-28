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
import networkx as nx

import nx_cugraph as nxcg
from nx_cugraph.utils import networkx_algorithm

__all__ = [
    "has_path",
]


@networkx_algorithm(version_added="24.04", _plc="bfs")
def has_path(G, source, target):
    # TODO PERF: make faster in core
    try:
        nxcg.bidirectional_shortest_path(G, source, target)
    except nx.NetworkXNoPath:
        return False
    return True
