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
import itertools

import cupy as cp
import networkx as nx
import pylibcugraph as plc

from nx_cugraph.convert import _to_undirected_graph
from nx_cugraph.utils import _groupby, networkx_algorithm, not_implemented_for

from ..isolate import _isolates

__all__ = [
    "number_connected_components",
    "connected_components",
    "is_connected",
    "node_connected_component",
]


@not_implemented_for("directed")
@networkx_algorithm
def number_connected_components(G):
    return sum(1 for _ in connected_components(G))
    # PREFERRED IMPLEMENTATION, BUT PLC DOES NOT HANDLE ISOLATED VERTICES WELL
    # G = _to_undirected_graph(G)
    # unused_node_ids, labels = plc.weakly_connected_components(
    #     resource_handle=plc.ResourceHandle(),
    #     graph=G._get_plc_graph(),
    #     offsets=None,
    #     indices=None,
    #     weights=None,
    #     labels=None,
    #     do_expensive_check=False,
    # )
    # return cp.unique(labels).size


@number_connected_components._can_run
def _(G):
    # NetworkX <= 3.2.1 does not check directedness for us
    try:
        return not G.is_directed()
    except Exception:
        return False


@not_implemented_for("directed")
@networkx_algorithm
def connected_components(G):
    G = _to_undirected_graph(G)
    if G.src_indices.size == 0:
        # TODO: PLC doesn't handle empty graphs (or isolated nodes) gracefully!
        return [{key} for key in G._nodeiter_to_iter(range(len(G)))]
    node_ids, labels = plc.weakly_connected_components(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        offsets=None,
        indices=None,
        weights=None,
        labels=None,
        do_expensive_check=False,
    )
    groups = _groupby(labels, node_ids)
    it = (G._nodearray_to_set(connected_ids) for connected_ids in groups.values())
    # TODO: PLC doesn't handle isolated vertices yet, so this is a temporary fix
    isolates = _isolates(G)
    if isolates.size > 0:
        isolates = isolates[isolates > node_ids.max()]
        if isolates.size > 0:
            it = itertools.chain(
                it, ({node} for node in G._nodearray_to_list(isolates))
            )
    return it


@not_implemented_for("directed")
@networkx_algorithm
def is_connected(G):
    G = _to_undirected_graph(G)
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept(
            "Connectivity is undefined for the null graph."
        )
    for community in connected_components(G):
        return len(community) == len(G)
    raise RuntimeError  # pragma: no cover
    # PREFERRED IMPLEMENTATION, BUT PLC DOES NOT HANDLE ISOLATED VERTICES WELL
    # unused_node_ids, labels = plc.weakly_connected_components(
    #     resource_handle=plc.ResourceHandle(),
    #     graph=G._get_plc_graph(),
    #     offsets=None,
    #     indices=None,
    #     weights=None,
    #     labels=None,
    #     do_expensive_check=False,
    # )
    # return labels.size == len(G) and cp.unique(labels).size == 1


@not_implemented_for("directed")
@networkx_algorithm
def node_connected_component(G, n):
    # We could also do plain BFS from n
    G = _to_undirected_graph(G)
    node_id = n if G.key_to_id is None else G.key_to_id[n]
    node_ids, labels = plc.weakly_connected_components(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        offsets=None,
        indices=None,
        weights=None,
        labels=None,
        do_expensive_check=False,
    )
    indices = cp.nonzero(node_ids == node_id)[0]
    if indices.size == 0:
        return {n}
    return G._nodearray_to_set(node_ids[labels == labels[indices[0]]])
