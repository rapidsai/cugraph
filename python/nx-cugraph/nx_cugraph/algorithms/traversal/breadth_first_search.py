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
from itertools import repeat

import cupy as cp
import networkx as nx
import numpy as np
import pylibcugraph as plc

import nx_cugraph as nxcg
from nx_cugraph import _nxver
from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import _groupby, index_dtype, networkx_algorithm

__all__ = [
    "bfs_edges",
    "bfs_tree",
    "bfs_predecessors",
    "bfs_successors",
    "descendants_at_distance",
    "bfs_layers",
    "generic_bfs_edges",
]


def _check_G_and_source(G, source):
    G = _to_graph(G)
    if source not in G:
        hash(source)  # To raise TypeError if appropriate
        raise nx.NetworkXError(
            f"The node {source} is not in the {G.__class__.__name__.lower()}."
        )
    return G


def _bfs(G, source, *, depth_limit=None, reverse=False):
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
    return distances[mask], predecessors[mask], node_ids[mask]


if _nxver < (3, 4):

    @networkx_algorithm(is_incomplete=True, version_added="24.02", _plc="bfs")
    def generic_bfs_edges(
        G, source, neighbors=None, depth_limit=None, sort_neighbors=None
    ):
        """`neighbors` and `sort_neighbors` parameters are not yet supported."""
        if neighbors is not None:
            raise NotImplementedError(
                "neighbors argument in generic_bfs_edges is not currently supported"
            )
        if sort_neighbors is not None:
            raise NotImplementedError(
                "sort_neighbors argument in generic_bfs_edges is not supported"
            )
        return bfs_edges(G, source, depth_limit=depth_limit)

    @generic_bfs_edges._can_run
    def _(G, source, neighbors=None, depth_limit=None, sort_neighbors=None):
        return neighbors is None and sort_neighbors is None

else:

    @networkx_algorithm(is_incomplete=True, version_added="24.02", _plc="bfs")
    def generic_bfs_edges(G, source, neighbors=None, depth_limit=None):
        """`neighbors` parameter is not yet supported."""
        if neighbors is not None:
            raise NotImplementedError(
                "neighbors argument in generic_bfs_edges is not currently supported"
            )
        return bfs_edges(G, source, depth_limit=depth_limit)

    @generic_bfs_edges._can_run
    def _(G, source, neighbors=None, depth_limit=None):
        return neighbors is None


@networkx_algorithm(is_incomplete=True, version_added="24.02", _plc="bfs")
def bfs_edges(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    """`sort_neighbors` parameter is not yet supported."""
    if sort_neighbors is not None:
        raise NotImplementedError(
            "sort_neighbors argument in bfs_edges is not currently supported"
        )
    G = _check_G_and_source(G, source)
    if depth_limit is not None and depth_limit < 1:
        return
    distances, predecessors, node_ids = _bfs(
        G, source, depth_limit=depth_limit, reverse=reverse
    )
    # Using groupby like this is similar to bfs_predecessors
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


@networkx_algorithm(is_incomplete=True, version_added="24.02", _plc="bfs")
def bfs_tree(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    """`sort_neighbors` parameter is not yet supported."""
    if sort_neighbors is not None:
        raise NotImplementedError(
            "sort_neighbors argument in bfs_tree is not currently supported"
        )
    is_compat_graph = isinstance(G, nxcg.Graph)
    G = _check_G_and_source(G, source)
    if depth_limit is not None and depth_limit < 1:
        return nxcg.CudaDiGraph.from_coo(
            1,
            cp.array([], dtype=index_dtype),
            cp.array([], dtype=index_dtype),
            id_to_key=[source],
            use_compat_graph=is_compat_graph,
        )

    distances, predecessors, node_ids = _bfs(
        G,
        source,
        depth_limit=depth_limit,
        reverse=reverse,
    )
    if predecessors.size == 0:
        return nxcg.CudaDiGraph.from_coo(
            1,
            cp.array([], dtype=index_dtype),
            cp.array([], dtype=index_dtype),
            id_to_key=[source],
            use_compat_graph=is_compat_graph,
        )
    # TODO: create renumbering helper function(s)
    unique_node_ids = cp.unique(cp.hstack((predecessors, node_ids)))
    # Renumber edges
    src_indices = cp.searchsorted(unique_node_ids, predecessors).astype(index_dtype)
    dst_indices = cp.searchsorted(unique_node_ids, node_ids).astype(index_dtype)
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
    return nxcg.CudaDiGraph.from_coo(
        unique_node_ids.size,
        src_indices,
        dst_indices,
        key_to_id=key_to_id,
        use_compat_graph=is_compat_graph,
    )


@bfs_tree._can_run
def _(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    return sort_neighbors is None


@networkx_algorithm(is_incomplete=True, version_added="24.02", _plc="bfs")
def bfs_successors(G, source, depth_limit=None, sort_neighbors=None):
    """`sort_neighbors` parameter is not yet supported."""
    if sort_neighbors is not None:
        raise NotImplementedError(
            "sort_neighbors argument in bfs_successors is not currently supported"
        )
    G = _check_G_and_source(G, source)
    if depth_limit is not None and depth_limit < 1:
        yield (source, [])
        return

    distances, predecessors, node_ids = _bfs(G, source, depth_limit=depth_limit)
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


@networkx_algorithm(version_added="24.02", _plc="bfs")
def bfs_layers(G, sources):
    G = _to_graph(G)
    if sources in G:
        sources = [sources]
    else:
        sources = set(sources)
        if not all(source in G for source in sources):
            node = next(source for source in sources if source not in G)
            raise nx.NetworkXError(f"The node {node} is not in the graph.")
        sources = list(sources)
    source_ids = G._list_to_nodearray(sources)
    distances, predecessors, node_ids = plc.bfs(
        handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        sources=source_ids,
        direction_optimizing=False,
        depth_limit=-1,
        compute_predecessors=False,
        do_expensive_check=False,
    )
    mask = distances != np.iinfo(distances.dtype).max
    distances = distances[mask]
    node_ids = node_ids[mask]
    groups = _groupby(distances, node_ids)
    return (G._nodearray_to_list(groups[key]) for key in range(len(groups)))


@networkx_algorithm(is_incomplete=True, version_added="24.02", _plc="bfs")
def bfs_predecessors(G, source, depth_limit=None, sort_neighbors=None):
    """`sort_neighbors` parameter is not yet supported."""
    if sort_neighbors is not None:
        raise NotImplementedError(
            "sort_neighbors argument in bfs_predecessors is not currently supported"
        )
    G = _check_G_and_source(G, source)
    if depth_limit is not None and depth_limit < 1:
        return

    distances, predecessors, node_ids = _bfs(G, source, depth_limit=depth_limit)
    # We include `predecessors` in the groupby for "nicer" iteration order
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


@networkx_algorithm(version_added="24.02", _plc="bfs")
def descendants_at_distance(G, source, distance):
    G = _check_G_and_source(G, source)
    if distance is None or distance < 0:
        return set()
    if distance == 0:
        return {source}

    src_index = source if G.key_to_id is None else G.key_to_id[source]
    distances, predecessors, node_ids = plc.bfs(
        handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        sources=cp.array([src_index], dtype=index_dtype),
        direction_optimizing=False,
        depth_limit=distance,
        compute_predecessors=False,
        do_expensive_check=False,
    )
    mask = distances == distance
    node_ids = node_ids[mask]
    return G._nodearray_to_set(node_ids)
