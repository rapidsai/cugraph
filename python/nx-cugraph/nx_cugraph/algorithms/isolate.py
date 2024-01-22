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

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import index_dtype, networkx_algorithm

if TYPE_CHECKING:  # pragma: no cover
    from nx_cugraph.typing import IndexValue

__all__ = ["is_isolate", "isolates", "number_of_isolates"]


@networkx_algorithm(version_added="23.10")
def is_isolate(G, n):
    G = _to_graph(G)
    index = n if G.key_to_id is None else G.key_to_id[n]
    return not (
        (G.src_indices == index).any().tolist()
        or G.is_directed()
        and (G.dst_indices == index).any().tolist()
    )


def _mark_isolates(G, symmetrize=None) -> cp.ndarray[bool]:
    """Return a boolean mask array indicating indices of isolated nodes."""
    mark_isolates = cp.ones(len(G), bool)
    if G.is_directed() and symmetrize == "intersection":
        N = G._N
        # Upcast to int64 so indices don't overflow
        src_dst = N * G.src_indices.astype(np.int64) + G.dst_indices
        src_dst_T = G.src_indices + N * G.dst_indices.astype(np.int64)
        src_dst_new = cp.intersect1d(src_dst, src_dst_T)
        new_indices = cp.floor_divide(src_dst_new, N, dtype=index_dtype)
        mark_isolates[new_indices] = False
    else:
        mark_isolates[G.src_indices] = False
        if G.is_directed():
            mark_isolates[G.dst_indices] = False
    return mark_isolates


def _isolates(G, symmetrize=None) -> cp.ndarray[IndexValue]:
    """Like isolates, but return an array of indices instead of an iterator of nodes."""
    G = _to_graph(G)
    return cp.nonzero(_mark_isolates(G, symmetrize=symmetrize))[0]


@networkx_algorithm(version_added="23.10")
def isolates(G):
    G = _to_graph(G)
    return G._nodeiter_to_iter(iter(_isolates(G).tolist()))


@networkx_algorithm(version_added="23.10")
def number_of_isolates(G):
    G = _to_graph(G)
    return _mark_isolates(G).sum().tolist()
