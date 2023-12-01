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
from itertools import repeat

import cupy as cp
import networkx as nx
import pylibcugraph as plc

import nx_cugraph as nxcg
from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import _groupby, index_dtype, networkx_algorithm

__all__ = [
    "bfs_edges",
    "bfs_tree",
    "bfs_predecessors",
    "bfs_successors",
    "generic_bfs_edges",
]


@networkx_algorithm
def generic_bfs_edges(G, source, neighbors=None, depth_limit=None, sort_neighbors=None):
    """`neighbors` parameter is not yet supported."""
    return bfs_edges(source, depth_limit=depth_limit)


@generic_bfs_edges._can_run
def _(G, source, neighbors=None, depth_limit=None, sort_neighbors=None):
    return neighbors is None and sort_neighbors is None


@networkx_algorithm
def bfs_edges(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    """`sort_neighbors` parameter is not yet supported."""
    # DRY warning: see also bfs_predecessors and bfs_tree
    G = _to_graph(G)
    if source not in G:
        hash(source)  # To raise TypeError if appropriate
        raise nx.NetworkXError(
            f"The node {source} is not in the {G.__class__.__name__.lower()}."
        )
    if depth_limit is not None and depth_limit < 1:
        return

    src_index = source if G.key_to_id is None else G.key_to_id[source]
    distances, predecessors, node_ids = plc.bfs(
        handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(switch_indices=reverse),
        sources=cp.array([src_index], dtype=index_dtype),
        direction_optimizing=False,
        depth_limit=-1 if depth_limit is None else depth_limit,
        compute_predecessors=True,
        do_expensive_check=False,
    )
    mask = predecessors >= 0
    distances = distances[mask]
    predecessors = predecessors[mask]
    node_ids = node_ids[mask]
    groups = _groupby([distances, predecessors], node_ids)
    id_to_key = G.id_to_key
    for key in sorted(groups):
        children_ids = groups[key]
        parent_id = key[1]
        parent = id_to_key[parent_id] if id_to_key is not None else parent_id
        yield from zip(
            repeat(parent, children_ids.size),
            G._nodeiter_to_iter(children_ids.tolist()),
        )


@bfs_edges._can_run
def _(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    return sort_neighbors is None


@networkx_algorithm
def bfs_tree(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    """`sort_neighbors` parameter is not yet supported."""
    # DRY warning: see also bfs_edges and bfs_predecessors
    G = _to_graph(G)
    if source not in G:
        hash(source)  # To raise TypeError if appropriate
        raise nx.NetworkXError(
            f"The node {source} is not in the {G.__class__.__name__.lower()}."
        )
    if depth_limit is not None and depth_limit < 1:
        return nxcg.DiGraph.from_coo(
            1,
            cp.array([], dtype=index_dtype),
            cp.array([], dtype=index_dtype),
            id_to_key=[source],
        )

    src_index = source if G.key_to_id is None else G.key_to_id[source]
    distances, predecessors, node_ids = plc.bfs(
        handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(switch_indices=reverse),
        sources=cp.array([src_index], dtype=index_dtype),
        direction_optimizing=False,
        depth_limit=-1 if depth_limit is None else depth_limit,
        compute_predecessors=True,
        do_expensive_check=False,
    )
    mask = predecessors >= 0
    predecessors = predecessors[mask]
    if predecessors.size == 0:
        return nxcg.DiGraph.from_coo(
            1,
            cp.array([], dtype=index_dtype),
            cp.array([], dtype=index_dtype),
            id_to_key=[source],
        )
    node_ids = node_ids[mask]
    # TODO: create renumbering helper function(s)
    unique_node_ids = cp.unique(cp.hstack((predecessors, node_ids)))
    # Renumber edges
    # Option 1
    src_indices = cp.searchsorted(unique_node_ids, predecessors)
    dst_indices = cp.searchsorted(unique_node_ids, node_ids)
    # Option 2
    # mapper = cp.zeros(len(G), index_dtype)
    # mapper[unique_node_ids] = cp.arange(unique_node_ids.size, dtype=mapper.dtype)
    # src_indices = mapper[predecessors]
    # dst_indices = mapper[node_ids]
    # Renumber nodes
    if (id_to_key := G.id_to_key) is not None:
        key_to_id = {
            id_to_key[old_index]: new_index
            for new_index, old_index in enumerate(unique_node_ids.tolist())
        }
    else:
        key_to_id = {
            old_index: new_index
            for new_index, old_index in enumerate(unique_node_ids.tolist())
        }
    return nxcg.DiGraph.from_coo(
        unique_node_ids.size,
        src_indices,
        dst_indices,
        key_to_id=key_to_id,
    )


@bfs_tree._can_run
def _(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    return sort_neighbors is None


@networkx_algorithm
def bfs_successors(G, source, depth_limit=None, sort_neighbors=None):
    """`sort_neighbors` parameter is not yet supported."""
    G = _to_graph(G)
    if source not in G:
        hash(source)  # To raise TypeError if appropriate
        raise nx.NetworkXError(
            f"The node {source} is not in the {G.__class__.__name__.lower()}."
        )
    if depth_limit is not None and depth_limit < 1:
        yield (source, [])
        return

    src_index = source if G.key_to_id is None else G.key_to_id[source]
    distances, predecessors, node_ids = plc.bfs(
        handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        sources=cp.array([src_index], dtype=index_dtype),
        direction_optimizing=False,
        depth_limit=-1 if depth_limit is None else depth_limit,
        compute_predecessors=True,
        do_expensive_check=False,
    )
    mask = predecessors >= 0
    distances = distances[mask]
    predecessors = predecessors[mask]
    node_ids = node_ids[mask]
    groups = _groupby([distances, predecessors], node_ids)
    id_to_key = G.id_to_key
    for key in sorted(groups):
        children_ids = groups[key]
        parent_id = key[1]
        parent = id_to_key[parent_id] if id_to_key is not None else parent_id
        children = G._nodearray_to_list(children_ids)
        yield (parent, children)


@bfs_successors._can_run
def _(G, source, depth_limit=None, sort_neighbors=None):
    return sort_neighbors is None


@networkx_algorithm
def bfs_predecessors(G, source, depth_limit=None, sort_neighbors=None):
    """`sort_neighbors` parameter is not yet supported."""
    # DRY warning: see also bfs_edges and bfs_tree
    G = _to_graph(G)
    if source not in G:
        hash(source)  # To raise TypeError if appropriate
        raise nx.NetworkXError(
            f"The node {source} is not in the {G.__class__.__name__.lower()}."
        )
    if depth_limit is not None and depth_limit < 1:
        return

    src_index = source if G.key_to_id is None else G.key_to_id[source]
    distances, predecessors, node_ids = plc.bfs(
        handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        sources=cp.array([src_index], dtype=index_dtype),
        direction_optimizing=False,
        depth_limit=-1 if depth_limit is None else depth_limit,
        compute_predecessors=True,
        do_expensive_check=False,
    )
    mask = predecessors >= 0
    distances = distances[mask]
    predecessors = predecessors[mask]
    node_ids = node_ids[mask]
    groups = _groupby([distances, predecessors], node_ids)
    id_to_key = G.id_to_key
    for key in sorted(groups):
        children_ids = groups[key]
        parent_id = key[1]
        parent = id_to_key[parent_id] if id_to_key is not None else parent_id
        yield from zip(
            G._nodeiter_to_iter(children_ids.tolist()),
            repeat(parent, children_ids.size),
        )


@bfs_predecessors._can_run
def _(G, source, depth_limit=None, sort_neighbors=None):
    return sort_neighbors is None
