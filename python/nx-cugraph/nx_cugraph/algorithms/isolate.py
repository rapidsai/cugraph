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

from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import networkx_algorithm

__all__ = ["isolates"]


def _isolates(G) -> cp.ndarray:
    G = _to_graph(G)
    mark_isolates = cp.ones(len(G), bool)
    mark_isolates[G.row_indices] = False
    if G.is_directed():
        mark_isolates[G.col_indices] = False
    return cp.nonzero(mark_isolates)[0]


@networkx_algorithm
def isolates(G):
    G = _to_graph(G)
    return G._nodeiter_to_iter(iter(_isolates(G).tolist()))
