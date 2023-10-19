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

import cupy as cp

from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import networkx_algorithm

if TYPE_CHECKING:  # pragma: no cover
    from nx_cugraph.typing import IndexValue

__all__ = ["is_isolate", "isolates", "number_of_isolates"]


@networkx_algorithm
def is_isolate(G, n):
    G = _to_graph(G)
    index = n if G.key_to_id is None else G.key_to_id[n]
    return not (
        (G.row_indices == index).any().tolist()
        or G.is_directed()
        and (G.col_indices == index).any().tolist()
    )


def _mark_isolates(G) -> cp.ndarray[bool]:
    """Return a boolean mask array indicating indices of isolated nodes."""
    mark_isolates = cp.ones(len(G), bool)
    mark_isolates[G.row_indices] = False
    if G.is_directed():
        mark_isolates[G.col_indices] = False
    return mark_isolates


def _isolates(G) -> cp.ndarray[IndexValue]:
    """Like isolates, but return an array of indices instead of an iterator of nodes."""
    G = _to_graph(G)
    return cp.nonzero(_mark_isolates(G))[0]


@networkx_algorithm
def isolates(G):
    G = _to_graph(G)
    return G._nodeiter_to_iter(iter(_isolates(G).tolist()))


@networkx_algorithm
def number_of_isolates(G):
    G = _to_graph(G)
    return _mark_isolates(G).sum().tolist()
