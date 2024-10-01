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
import math

import cupy as cp
import networkx as nx
import numpy as np
import pylibcugraph as plc

import nx_cugraph as nxcg

from ..utils import _dtype_param, _get_float_dtype, index_dtype, networkx_algorithm

__all__ = ["ego_graph"]


@networkx_algorithm(
    extra_params=_dtype_param, version_added="24.06", _plc={"bfs", "ego_graph", "sssp"}
)
def ego_graph(
    G, n, radius=1, center=True, undirected=False, distance=None, *, dtype=None
):
    """Weighted ego_graph with negative cycles is not yet supported. `NotImplementedError` will be raised if there are negative `distance` edge weights."""  # noqa: E501
    if isinstance(G, nx.Graph):
        is_compat_graph = isinstance(G, nxcg.Graph)
        G = nxcg.from_networkx(G, preserve_all_attrs=True)
    else:
        is_compat_graph = False
    if n not in G:
        if distance is None:
            raise nx.NodeNotFound(f"Source {n} is not in G")
        raise nx.NodeNotFound(f"Node {n} not found in graph")
    src_index = n if G.key_to_id is None else G.key_to_id[n]
    symmetrize = "union" if undirected and G.is_directed() else None
    if distance is None or distance not in G.edge_values:
        # Simple BFS to determine nodes
        if radius is not None and radius <= 0:
            if center:
                node_ids = cp.array([src_index], dtype=index_dtype)
            else:
                node_ids = cp.empty(0, dtype=index_dtype)
            node_mask = None
        else:
            if radius is None or np.isinf(radius):
                radius = -1
            else:
                radius = math.ceil(radius)
            distances, unused_predecessors, node_ids = plc.bfs(
                handle=plc.ResourceHandle(),
                graph=G._get_plc_graph(symmetrize=symmetrize),
                sources=cp.array([src_index], index_dtype),
                direction_optimizing=False,  # True for undirected only; what's best?
                depth_limit=radius,
                compute_predecessors=False,
                do_expensive_check=False,
            )
            node_mask = distances != np.iinfo(distances.dtype).max
    else:
        # SSSP to determine nodes
        if callable(distance):
            raise NotImplementedError("callable `distance` argument is not supported")
        if symmetrize and G.is_multigraph():
            # G._get_plc_graph does not implement `symmetrize=True` w/ edge array
            raise NotImplementedError(
                "Weighted ego_graph with undirected=True not implemented"
            )
        # Check for negative values since we don't support negative cycles
        edge_vals = G.edge_values[distance]
        if distance in G.edge_masks:
            edge_vals = edge_vals[G.edge_masks[distance]]
        if (edge_vals < 0).any():
            raise NotImplementedError(
                "Negative edge weights not yet supported by ego_graph"
            )
        # PERF: we could use BFS if all edges are equal
        if radius is None:
            radius = np.inf
        dtype = _get_float_dtype(dtype, graph=G, weight=distance)
        node_ids, distances, unused_predecessors = plc.sssp(
            resource_handle=plc.ResourceHandle(),
            graph=(G.to_undirected() if symmetrize else G)._get_plc_graph(
                distance, 1, dtype
            ),
            source=src_index,
            cutoff=np.nextafter(radius, np.inf, dtype=np.float64),
            compute_predecessors=True,  # TODO: False is not yet supported
            do_expensive_check=False,
        )
        node_mask = distances != np.finfo(distances.dtype).max

    if node_mask is not None:
        if not center:
            node_mask &= node_ids != src_index
        node_ids = node_ids[node_mask]
    if node_ids.size == G._N:
        rv = G.copy()
        if is_compat_graph:
            return rv._to_compat_graph()
        return rv
    # TODO: create renumbering helper function(s)
    node_ids.sort()  # TODO: is this ever necessary? Keep for safety
    node_values = {key: val[node_ids] for key, val in G.node_values.items()}
    node_masks = {key: val[node_ids] for key, val in G.node_masks.items()}

    G._sort_edge_indices()  # TODO: is this ever necessary? Keep for safety
    edge_mask = cp.isin(G.src_indices, node_ids) & cp.isin(G.dst_indices, node_ids)
    src_indices = cp.searchsorted(node_ids, G.src_indices[edge_mask]).astype(
        index_dtype
    )
    dst_indices = cp.searchsorted(node_ids, G.dst_indices[edge_mask]).astype(
        index_dtype
    )
    edge_values = {key: val[edge_mask] for key, val in G.edge_values.items()}
    edge_masks = {key: val[edge_mask] for key, val in G.edge_masks.items()}

    # Renumber nodes
    if (id_to_key := G.id_to_key) is not None:
        key_to_id = {
            id_to_key[old_index]: new_index
            for new_index, old_index in enumerate(node_ids.tolist())
        }
    else:
        key_to_id = {
            old_index: new_index
            for new_index, old_index in enumerate(node_ids.tolist())
        }
    kwargs = {
        "N": node_ids.size,
        "src_indices": src_indices,
        "dst_indices": dst_indices,
        "edge_values": edge_values,
        "edge_masks": edge_masks,
        "node_values": node_values,
        "node_masks": node_masks,
        "key_to_id": key_to_id,
        "use_compat_graph": False,
    }
    if G.is_multigraph():
        if G.edge_keys is not None:
            kwargs["edge_keys"] = [
                x for x, m in zip(G.edge_keys, edge_mask.tolist()) if m
            ]
        if G.edge_indices is not None:
            kwargs["edge_indices"] = G.edge_indices[edge_mask]
    rv = G.__class__.from_coo(**kwargs)
    rv.graph.update(G.graph)
    if is_compat_graph:
        return rv._to_compat_graph()
    return rv


@ego_graph._can_run
def _(G, n, radius=1, center=True, undirected=False, distance=None, *, dtype=None):
    if distance is not None and undirected and G.is_directed() and G.is_multigraph():
        return "Weighted ego_graph with undirected=True not implemented"
    if distance is not None and nx.is_negatively_weighted(G, weight=distance):
        return "Weighted ego_graph with negative cycles not yet supported"
    if callable(distance):
        return "callable `distance` argument is not supported"
    return True
