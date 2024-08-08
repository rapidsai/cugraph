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
import networkx as nx
import numpy as np

import nx_cugraph as nxcg
from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import index_dtype, networkx_algorithm

__all__ = ["complement", "reverse"]


@networkx_algorithm(version_added="24.02")
def complement(G):
    G = _to_graph(G)
    N = G._N
    # Upcast to int64 so indices don't overflow.
    edges_a_b = N * G.src_indices.astype(np.int64) + G.dst_indices
    # Now compute flattened indices for all edges except self-loops
    # Alt (slower):
    # edges_full = np.arange(N * N)
    # edges_full = edges_full[(edges_full % (N + 1)).astype(bool)]
    edges_full = cp.arange(1, N * (N - 1) + 1) + cp.repeat(cp.arange(N - 1), N)
    edges_comp = cp.setdiff1d(
        edges_full,
        edges_a_b,
        assume_unique=not G.is_multigraph(),
    )
    src_indices, dst_indices = cp.divmod(edges_comp, N)
    return G.__class__.from_coo(
        N,
        src_indices.astype(index_dtype),
        dst_indices.astype(index_dtype),
        key_to_id=G.key_to_id,
    )


@networkx_algorithm(version_added="24.02")
def reverse(G, copy=True):
    if not G.is_directed():
        raise nx.NetworkXError("Cannot reverse an undirected graph.")
    if isinstance(G, nx.Graph):
        if not copy:
            raise RuntimeError(
                "Using `copy=False` is invalid when using a NetworkX graph "
                "as input to `nx_cugraph.reverse`"
            )
        G = nxcg.from_networkx(G, preserve_all_attrs=True)
    return G.reverse(copy=copy)
