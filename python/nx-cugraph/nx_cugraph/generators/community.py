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
import cupy as cp

import nx_cugraph as nxcg

from ..utils import networkx_algorithm
from ._utils import (
    _common_small_graph,
    _complete_graph_indices,
    _ensure_int,
    _ensure_nonnegative_int,
)

__all__ = [
    "caveman_graph",
]


@networkx_algorithm
def caveman_graph(l, k):  # noqa: E741
    l = _ensure_int(l)  # noqa: E741
    k = _ensure_int(k)
    N = _ensure_nonnegative_int(k * l)
    if l == 0 or k < 1:
        return _common_small_graph(N, None, None)
    k = _ensure_nonnegative_int(k)
    src_clique, dst_clique = _complete_graph_indices(k)
    src_cliques = [src_clique]
    dst_cliques = [dst_clique]
    src_cliques.extend(src_clique + i * k for i in range(1, l))
    dst_cliques.extend(dst_clique + i * k for i in range(1, l))
    src_indices = cp.hstack(src_cliques)
    dst_indices = cp.hstack(dst_cliques)
    return nxcg.Graph.from_coo(l * k, src_indices, dst_indices)
