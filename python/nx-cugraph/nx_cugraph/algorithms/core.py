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
import networkx as nx
import numpy as np
import pylibcugraph as plc

import nx_cugraph as nxcg
from nx_cugraph.utils import networkx_algorithm, not_implemented_for

__all__ = ["k_truss"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@networkx_algorithm
def k_truss(G, k):
    if is_nx := isinstance(G, nx.Graph):
        G = nxcg.from_networkx(G, preserve_all_attrs=True)
    if nxcg.number_of_selfloops(G) > 0:
        raise nx.NetworkXError(
            "Input graph has self loops which is not permitted; "
            "Consider using G.remove_edges_from(nx.selfloop_edges(G))."
        )
    # TODO: create renumbering helper function(s)
    if k < 3:
        # k-truss graph is comprised of nodes incident on k-2 triangles, so k<3 is a
        # boundary condition. Here, all we need to do is drop nodes with zero degree.
        # Technically, it would be okay to delete this branch of code, because
        # plc.k_truss_subgraph behaves the same for 0 <= k < 3. We keep this branch b/c
        # it's faster and has an "early return" if there are no nodes with zero degree.
        degrees = G._degrees_array()
        # Renumber step 0: node indices
        node_indices = degrees.nonzero()[0]
        if degrees.size == node_indices.size:
            # No change
            return G if is_nx else G.copy()
        src_indices = G.src_indices
        dst_indices = G.dst_indices
        # Renumber step 1: edge values (no changes needed)
        edge_values = {key: val.copy() for key, val in G.edge_values.items()}
        edge_masks = {key: val.copy() for key, val in G.edge_masks.items()}
    else:
        # int dtype for edge_indices would be preferred
        edge_indices = cp.arange(G.src_indices.size, dtype=np.float64)
        src_indices, dst_indices, edge_indices, _ = plc.k_truss_subgraph(
            resource_handle=plc.ResourceHandle(),
            graph=G._get_plc_graph(edge_array=edge_indices),
            k=k,
            do_expensive_check=False,
        )
        # Renumber step 0: node indices
        node_indices = cp.unique(cp.concatenate([src_indices, dst_indices]))
        # Renumber step 1: edge values
        edge_indices = edge_indices.astype(np.int64)
        edge_values = {key: val[edge_indices] for key, val in G.edge_values.items()}
        edge_masks = {key: val[edge_indices] for key, val in G.edge_masks.items()}
    # Renumber step 2: edge indices
    mapper = cp.zeros(len(G), src_indices.dtype)
    mapper[node_indices] = cp.arange(node_indices.size, dtype=mapper.dtype)
    src_indices = mapper[src_indices]
    dst_indices = mapper[dst_indices]
    # Renumber step 3: node values
    node_values = {key: val[node_indices] for key, val in G.node_values.items()}
    node_masks = {key: val[node_indices] for key, val in G.node_masks.items()}
    # Renumber step 4: key_to_id
    if (id_to_key := G.id_to_key) is not None:
        key_to_id = {
            id_to_key[old_index]: new_index
            for new_index, old_index in enumerate(node_indices.tolist())
        }
    else:
        key_to_id = None
    # Same as calling `G.from_coo`, but use __class__ to indicate it's a classmethod.
    new_graph = G.__class__.from_coo(
        node_indices.size,
        src_indices,
        dst_indices,
        edge_values,
        edge_masks,
        node_values,
        node_masks,
        key_to_id=key_to_id,
    )
    new_graph.graph.update(G.graph)
    return new_graph
