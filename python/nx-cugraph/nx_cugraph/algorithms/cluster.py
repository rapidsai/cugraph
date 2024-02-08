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
import pylibcugraph as plc

from nx_cugraph.convert import _to_undirected_graph
from nx_cugraph.utils import networkx_algorithm, not_implemented_for

__all__ = [
    "triangles",
    "average_clustering",
    "clustering",
    "transitivity",
]


def _triangles(G, nodes, symmetrize=None):
    if nodes is not None:
        if is_single_node := (nodes in G):
            nodes = [nodes if G.key_to_id is None else G.key_to_id[nodes]]
        else:
            nodes = list(nodes)
        nodes = G._list_to_nodearray(nodes)
    else:
        is_single_node = False
    if len(G) == 0:
        return None, None, is_single_node
    node_ids, triangles = plc.triangle_count(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(symmetrize=symmetrize),
        start_list=nodes,
        do_expensive_check=False,
    )
    return node_ids, triangles, is_single_node


@not_implemented_for("directed")
@networkx_algorithm(version_added="24.02", _plc="triangle_count")
def triangles(G, nodes=None):
    G = _to_undirected_graph(G)
    node_ids, triangles, is_single_node = _triangles(G, nodes)
    if len(G) == 0:
        return {}
    if is_single_node:
        return int(triangles[0])
    return G._nodearrays_to_dict(node_ids, triangles)


@not_implemented_for("directed")
@networkx_algorithm(is_incomplete=True, version_added="24.02", _plc="triangle_count")
def clustering(G, nodes=None, weight=None):
    """Directed graphs and `weight` parameter are not yet supported."""
    if weight is not None:
        raise NotImplementedError(
            "Weighted implementation of clustering not currently supported"
        )
    G = _to_undirected_graph(G)
    node_ids, triangles, is_single_node = _triangles(G, nodes)
    if len(G) == 0:
        return {}
    if is_single_node:
        numer = int(triangles[0])
        if numer == 0:
            return 0
        degree = int((G.src_indices == nodes).sum())
        return 2 * numer / (degree * (degree - 1))
    degrees = G._degrees_array(ignore_selfloops=True)[node_ids]
    denom = degrees * (degrees - 1)
    results = 2 * triangles / denom
    results = cp.where(denom, results, 0)  # 0 where we divided by 0
    return G._nodearrays_to_dict(node_ids, results)


@clustering._can_run
def _(G, nodes=None, weight=None):
    return weight is None and not G.is_directed()


@not_implemented_for("directed")
@networkx_algorithm(is_incomplete=True, version_added="24.02", _plc="triangle_count")
def average_clustering(G, nodes=None, weight=None, count_zeros=True):
    """Directed graphs and `weight` parameter are not yet supported."""
    if weight is not None:
        raise NotImplementedError(
            "Weighted implementation of average_clustering not currently supported"
        )
    G = _to_undirected_graph(G)
    node_ids, triangles, is_single_node = _triangles(G, nodes)
    if len(G) == 0:
        raise ZeroDivisionError
    degrees = G._degrees_array(ignore_selfloops=True)[node_ids]
    if not count_zeros:
        mask = triangles != 0
        triangles = triangles[mask]
        if triangles.size == 0:
            raise ZeroDivisionError
        degrees = degrees[mask]
    denom = degrees * (degrees - 1)
    results = 2 * triangles / denom
    if count_zeros:
        results = cp.where(denom, results, 0)  # 0 where we divided by 0
    return float(results.mean())


@average_clustering._can_run
def _(G, nodes=None, weight=None, count_zeros=True):
    return weight is None and not G.is_directed()


@not_implemented_for("directed")
@networkx_algorithm(is_incomplete=True, version_added="24.02", _plc="triangle_count")
def transitivity(G):
    """Directed graphs are not yet supported."""
    G = _to_undirected_graph(G)
    if len(G) == 0:
        return 0
    node_ids, triangles = plc.triangle_count(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        start_list=None,
        do_expensive_check=False,
    )
    numer = int(triangles.sum())
    if numer == 0:
        return 0
    degrees = G._degrees_array(ignore_selfloops=True)[node_ids]
    denom = int((degrees * (degrees - 1)).sum())
    return 2 * numer / denom


@transitivity._can_run
def _(G):
    # Is transitivity supposed to work on directed graphs?
    return not G.is_directed()
