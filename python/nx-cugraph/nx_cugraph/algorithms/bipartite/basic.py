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

from nx_cugraph.algorithms.cluster import _triangles
from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import networkx_algorithm

__all__ = [
    "is_bipartite",
]


@networkx_algorithm(plc="triangle_count", version_added="24.02")
def is_bipartite(G):
    G = _to_graph(G)
    # Counting triangles may not be the fastest way to do this, but it is simple.
    node_ids, triangles, is_single_node = _triangles(
        G, None, symmetrize="union" if G.is_directed() else None
    )
    return int(cp.count_nonzero(triangles)) == 0
