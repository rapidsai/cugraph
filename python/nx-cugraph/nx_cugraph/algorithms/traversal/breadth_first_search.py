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
import pylibcugraph as plc

from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import _groupby, index_dtype, networkx_algorithm

__all__ = [
    "bfs_predecessors",
    "bfs_successors",
]


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
    groups = _groupby(distances, [predecessors, node_ids])
    id_to_key = G.id_to_key
    for key in range(1, len(groups) + 1):
        parent_ids, children_ids = groups[key]
        parent_id = parent_ids[0].tolist()
        parent = id_to_key[parent_id] if id_to_key is not None else parent_id
        children = G._nodearray_to_list(children_ids)
        yield (parent, children)


@bfs_successors._can_run
def _(G, source, depth_limit=None, sort_neighbors=None):
    return sort_neighbors is None


@networkx_algorithm
def bfs_predecessors(G, source, depth_limit=None, sort_neighbors=None):
    """`sort_neighbors` parameter is not yet supported."""
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
    groups = _groupby(distances, [predecessors, node_ids])
    for key in range(1, len(groups) + 1):
        parent_ids, children_ids = groups[key]
        yield from zip(
            G._nodeiter_to_iter(children_ids.tolist()),
            G._nodeiter_to_iter(parent_ids.tolist()),
        )


@bfs_predecessors._can_run
def _(G, source, depth_limit=None, sort_neighbors=None):
    return sort_neighbors is None
