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
import itertools

import cupy as cp
import networkx as nx
import numpy as np
import pylibcugraph as plc

from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import _groupby, index_dtype, networkx_algorithm

__all__ = [
    "bidirectional_shortest_path",
    "single_source_shortest_path",
    "single_source_shortest_path_length",
    "single_target_shortest_path",
    "single_target_shortest_path_length",
    "all_pairs_shortest_path",
    "all_pairs_shortest_path_length",
]

concat = itertools.chain.from_iterable


@networkx_algorithm(version_added="23.12", _plc="bfs")
def single_source_shortest_path_length(G, source, cutoff=None):
    return _single_shortest_path_length(G, source, cutoff, "Source")


@networkx_algorithm(version_added="23.12", _plc="bfs")
def single_target_shortest_path_length(G, target, cutoff=None):
    return _single_shortest_path_length(G, target, cutoff, "Target")


def _single_shortest_path_length(G, source, cutoff, kind):
    G = _to_graph(G)
    if source not in G:
        raise nx.NodeNotFound(f"{kind} {source} is not in G")
    if G.src_indices.size == 0:
        return {source: 0}
    if cutoff is None:
        cutoff = -1
    elif cutoff <= 0:
        return {source: 0}
    src_index = source if G.key_to_id is None else G.key_to_id[source]
    distances, predecessors, node_ids = plc.bfs(
        handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(switch_indices=kind == "Target"),
        sources=cp.array([src_index], index_dtype),
        direction_optimizing=False,  # True for undirected only; what's recommended?
        depth_limit=cutoff,
        compute_predecessors=False,
        do_expensive_check=False,
    )
    mask = distances != np.iinfo(distances.dtype).max
    return G._nodearrays_to_dict(node_ids[mask], distances[mask])


@networkx_algorithm(version_added="24.04", _plc="bfs")
def all_pairs_shortest_path_length(G, cutoff=None):
    # TODO PERF: batched bfs to compute many at once
    for n in G:
        yield (n, single_source_shortest_path_length(G, n, cutoff))


@networkx_algorithm(version_added="24.04", _plc="bfs")
def bidirectional_shortest_path(G, source, target):
    # TODO PERF: do bidirectional traversal in core
    G = _to_graph(G)
    if source not in G or target not in G:
        raise nx.NodeNotFound(f"Either source {source} or target {target} is not in G")
    src_index = source if G.key_to_id is None else G.key_to_id[source]
    dst_index = target if G.key_to_id is None else G.key_to_id[target]
    if src_index == dst_index:
        return [source]
    distances, predecessors, node_ids = plc.bfs(
        handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        sources=cp.array([src_index], index_dtype),
        direction_optimizing=False,  # True for undirected only; what's recommended?
        depth_limit=-1,
        compute_predecessors=True,
        do_expensive_check=False,
    )
    mask = distances != np.iinfo(distances.dtype).max
    predecessors = predecessors[mask]
    node_ids = node_ids[mask]
    d = dict(zip(node_ids.tolist(), predecessors.tolist()))
    if dst_index not in d:
        raise nx.NetworkXNoPath(f"No path between {source} and {target}.")
    # Consider making utility functions for creating paths
    cur = dst_index
    path = [dst_index]
    while cur != src_index:
        cur = d[cur]
        path.append(cur)
    if (id_to_key := G.id_to_key) is not None:
        path = [id_to_key[cur] for cur in reversed(path)]
    else:
        path.reverse()
    return path


@networkx_algorithm(version_added="24.04", _plc="bfs")
def single_source_shortest_path(G, source, cutoff=None):
    return _single_shortest_path(G, source, cutoff, "Source", True)


@networkx_algorithm(version_added="24.04", _plc="bfs")
def single_target_shortest_path(G, target, cutoff=None):
    return _single_shortest_path(G, target, cutoff, "Target", False)


def _single_shortest_path(G, source, cutoff, kind, reverse_path):
    G = _to_graph(G)
    if source not in G:
        raise nx.NodeNotFound(f"{kind} {source} not in G")
    if G.src_indices.size == 0:
        return {source: [source]}
    if cutoff is None:
        cutoff = -1
    elif cutoff <= 0:
        return {source: [source]}
    src_index = source if G.key_to_id is None else G.key_to_id[source]
    distances, predecessors, node_ids = plc.bfs(
        handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(switch_indices=kind == "Target"),
        sources=cp.array([src_index], index_dtype),
        direction_optimizing=False,  # True for undirected only; what's recommended?
        depth_limit=cutoff,
        compute_predecessors=True,
        do_expensive_check=False,
    )
    mask = distances != np.iinfo(distances.dtype).max
    groups = _groupby(distances[mask], [predecessors[mask], node_ids[mask]])

    # `pred_node_iter` does the equivalent as these nested for loops:
    # for distance in range(1, len(groups)):
    #     preds, nodes = groups[distance]
    #     for pred, node in zip(preds.tolist(), nodes.tolist()):
    if G.key_to_id is None:
        pred_node_iter = concat(
            zip(*(x.tolist() for x in groups[distance]))
            for distance in range(1, len(groups))
        )
    else:
        pred_node_iter = concat(
            zip(*(G._nodeiter_to_iter(x.tolist()) for x in groups[distance]))
            for distance in range(1, len(groups))
        )
    # Consider making utility functions for creating paths
    paths = {source: [source]}
    if reverse_path:
        for pred, node in pred_node_iter:
            paths[node] = [*paths[pred], node]
    else:
        for pred, node in pred_node_iter:
            paths[node] = [node, *paths[pred]]
    return paths


@networkx_algorithm(version_added="24.04", _plc="bfs")
def all_pairs_shortest_path(G, cutoff=None):
    # TODO PERF: batched bfs to compute many at once
    for n in G:
        yield (n, single_source_shortest_path(G, n, cutoff))
