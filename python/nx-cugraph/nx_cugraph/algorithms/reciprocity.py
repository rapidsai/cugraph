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
import cupy as cp
import networkx as nx
import numpy as np

from nx_cugraph.convert import _to_directed_graph
from nx_cugraph.utils import networkx_algorithm, not_implemented_for

__all__ = ["reciprocity", "overall_reciprocity"]


@not_implemented_for("undirected", "multigraph")
@networkx_algorithm(version_added="24.02")
def reciprocity(G, nodes=None):
    if nodes is None:
        return overall_reciprocity(G)
    G = _to_directed_graph(G)
    N = G._N
    # 'nodes' can also be a single node identifier
    if nodes in G:
        index = nodes if G.key_to_id is None else G.key_to_id[nodes]
        mask = (G.src_indices == index) | (G.dst_indices == index)
        src_indices = G.src_indices[mask]
        if src_indices.size == 0:
            raise nx.NetworkXError("Not defined for isolated nodes.")
        dst_indices = G.dst_indices[mask]
        # Create two lists of edge identifiers, one for each direction.
        # Edge identifiers can be created from a pair of node
        # identifiers. Simply adding src IDs to dst IDs is not adequate, so
        # make one set of values (either src or dst depending on direction)
        # unique by multiplying values by N.
        # Upcast to int64 so indices don't overflow.
        edges_a_b = N * src_indices.astype(np.int64) + dst_indices
        edges_b_a = src_indices + N * dst_indices.astype(np.int64)
        # Find the matching edge identifiers in each list. The edge identifier
        # generation ensures the ID for A->B == the ID for B->A
        recip_indices = cp.intersect1d(
            edges_a_b,
            edges_b_a,
            # assume_unique=True,  # cupy <= 12.2.0 also assumes sorted
        )
        num_selfloops = (src_indices == dst_indices).sum().tolist()
        return (recip_indices.size - num_selfloops) / edges_a_b.size

    # Don't include self-loops
    mask = G.src_indices != G.dst_indices
    src_indices = G.src_indices[mask]
    dst_indices = G.dst_indices[mask]
    # Create two lists of edges, one for each direction, and find the matching
    # IDs in each list (see description above).
    edges_a_b = N * src_indices.astype(np.int64) + dst_indices
    edges_b_a = src_indices + N * dst_indices.astype(np.int64)
    recip_indices = cp.intersect1d(
        edges_a_b,
        edges_b_a,
        # assume_unique=True,  # cupy <= 12.2.0 also assumes sorted
    )
    numer = cp.bincount(recip_indices // N, minlength=N)
    denom = cp.bincount(src_indices, minlength=N)
    denom += cp.bincount(dst_indices, minlength=N)
    recip = 2 * numer / denom
    node_ids = G._nodekeys_to_nodearray(nodes)
    return G._nodearrays_to_dict(node_ids, recip[node_ids])


@not_implemented_for("undirected", "multigraph")
@networkx_algorithm(version_added="24.02")
def overall_reciprocity(G):
    G = _to_directed_graph(G)
    if G.number_of_edges() == 0:
        raise nx.NetworkXError("Not defined for empty graphs")
    # Create two lists of edges, one for each direction, and find the matching
    # IDs in each list (see description in reciprocity()).
    edges_a_b = G._N * G.src_indices.astype(np.int64) + G.dst_indices
    edges_b_a = G.src_indices + G._N * G.dst_indices.astype(np.int64)
    recip_indices = cp.intersect1d(
        edges_a_b,
        edges_b_a,
        # assume_unique=True,  # cupy <= 12.2.0 also assumes sorted
    )
    num_selfloops = (G.src_indices == G.dst_indices).sum().tolist()
    return (recip_indices.size - num_selfloops) / edges_a_b.size
