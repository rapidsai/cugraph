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

from nx_cugraph import _nxver
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
    G = _to_graph(G)
    return _bfs(G, source, cutoff, "Source", return_type="length")


@networkx_algorithm(version_added="23.12", _plc="bfs")
def single_target_shortest_path_length(G, target, cutoff=None):
    G = _to_graph(G)
    rv = _bfs(G, target, cutoff, "Target", return_type="length")
    if _nxver <= (3, 4):
        return iter(rv.items())
    return rv


@networkx_algorithm(version_added="24.04", _plc="bfs")
def all_pairs_shortest_path_length(G, cutoff=None):
    # TODO PERF: batched bfs to compute many at once
    G = _to_graph(G)
    for n in G:
        yield (n, _bfs(G, n, cutoff, "Source", return_type="length"))


@networkx_algorithm(version_added="24.04", _plc="bfs")
def bidirectional_shortest_path(G, source, target):
    # TODO PERF: do bidirectional traversal in core
    G = _to_graph(G)
    if source not in G or target not in G:
        if _nxver <= (3, 3):
            raise nx.NodeNotFound(
                f"Either source {source} or target {target} is not in G"
            )
        missing = f"Source {source}" if source not in G else f"Target {target}"
        raise nx.NodeNotFound(f"{missing} is not in G")
    return _bfs(G, source, None, "Source", return_type="path", target=target)


@networkx_algorithm(version_added="24.04", _plc="bfs")
def single_source_shortest_path(G, source, cutoff=None):
    G = _to_graph(G)
    return _bfs(G, source, cutoff, "Source", return_type="path")


@networkx_algorithm(version_added="24.04", _plc="bfs")
def single_target_shortest_path(G, target, cutoff=None):
    G = _to_graph(G)
    return _bfs(G, target, cutoff, "Target", return_type="path", reverse_path=True)


@networkx_algorithm(version_added="24.04", _plc="bfs")
def all_pairs_shortest_path(G, cutoff=None):
    # TODO PERF: batched bfs to compute many at once
    G = _to_graph(G)
    for n in G:
        yield (n, _bfs(G, n, cutoff, "Source", return_type="path"))


def _bfs(
    G, source, cutoff, kind, *, return_type, reverse_path=False, target=None, scale=None
):
    """BFS for unweighted shortest path algorithms.

    Parameters
    ----------
    source: node label

    cutoff: int, optional

    kind: {"Source", "Target"}

    return_type: {"length", "path", "length-path"}

    reverse_path: bool

    target: node label

    scale: int or float, optional
        The amount to scale the lengths
    """
    # DRY: _sssp in weighted.py has similar code
    if source not in G:
        # Different message to pass networkx tests
        if return_type == "length":
            raise nx.NodeNotFound(f"{kind} {source} is not in G")
        raise nx.NodeNotFound(f"{kind} {source} not in G")
    if target is not None:
        if source == target or cutoff is not None and cutoff <= 0:
            if return_type == "path":
                return [source]
            if return_type == "length":
                return 0
            # return_type == "length-path"
            return 0, [source]
        if target not in G or G.src_indices.size == 0:
            raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")
    elif G.src_indices.size == 0 or cutoff is not None and cutoff <= 0:
        if return_type == "path":
            return {source: [source]}
        if return_type == "length":
            return {source: 0}
        # return_type == "length-path"
        return {source: 0}, {source: [source]}

    if cutoff is None or np.isinf(cutoff):
        cutoff = -1
    src_index = source if G.key_to_id is None else G.key_to_id[source]
    distances, predecessors, node_ids = plc.bfs(
        handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(switch_indices=kind == "Target"),
        sources=cp.array([src_index], index_dtype),
        direction_optimizing=False,  # True for undirected only; what's recommended?
        depth_limit=cutoff,
        compute_predecessors=return_type != "length",
        do_expensive_check=False,
    )
    mask = distances != np.iinfo(distances.dtype).max
    node_ids = node_ids[mask]
    if return_type != "path":
        lengths = distances = distances[mask]
        if scale is not None:
            lengths = scale * lengths
        lengths = G._nodearrays_to_dict(node_ids, lengths)
        if target is not None:
            if target not in lengths:
                raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")
            lengths = lengths[target]
    if return_type != "length":
        if target is not None:
            d = dict(zip(node_ids.tolist(), predecessors[mask].tolist()))
            dst_index = target if G.key_to_id is None else G.key_to_id[target]
            if dst_index not in d:
                raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")
            cur = dst_index
            paths = [dst_index]
            while cur != src_index:
                cur = d[cur]
                paths.append(cur)
            if (id_to_key := G.id_to_key) is not None:
                if reverse_path:
                    paths = [id_to_key[cur] for cur in paths]
                else:
                    paths = [id_to_key[cur] for cur in reversed(paths)]
            elif not reverse_path:
                paths.reverse()
        else:
            if return_type == "path":
                distances = distances[mask]
            groups = _groupby(distances, [predecessors[mask], node_ids])

            # `pred_node_iter` does the equivalent as these nested for loops:
            # for length in range(1, len(groups)):
            #     preds, nodes = groups[length]
            #     for pred, node in zip(preds.tolist(), nodes.tolist()):
            if G.key_to_id is None:
                pred_node_iter = concat(
                    zip(*(x.tolist() for x in groups[length]))
                    for length in range(1, len(groups))
                )
            else:
                pred_node_iter = concat(
                    zip(*(G._nodeiter_to_iter(x.tolist()) for x in groups[length]))
                    for length in range(1, len(groups))
                )
            # Consider making utility functions for creating paths
            paths = {source: [source]}
            if reverse_path:
                for pred, node in pred_node_iter:
                    paths[node] = [node, *paths[pred]]
            else:
                for pred, node in pred_node_iter:
                    paths[node] = [*paths[pred], node]
    if return_type == "path":
        return paths
    if return_type == "length":
        return lengths
    # return_type == "length-path"
    return lengths, paths
