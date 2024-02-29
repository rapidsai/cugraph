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
import networkx as nx
import numpy as np
import pylibcugraph as plc

from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import (
    _dtype_param,
    _get_float_dtype,
    _groupby,
    networkx_algorithm,
)

__all__ = [
    # "bellman_ford_path",
    # "bellman_ford_path_length",
    # "single_source_bellman_ford",
    "single_source_bellman_ford_path",
    "single_source_bellman_ford_path_length",
    # "all_pairs_bellman_ford_path",
    # "all_pairs_bellman_ford_path_length",
    # "bellman_ford_predecessor_and_distance",
]


@networkx_algorithm(extra_params=_dtype_param, version_added="24.04", _plc="sssp")
def single_source_bellman_ford_path_length(G, source, weight="weight", *, dtype=None):
    """Negative cycles are not yet supported!"""
    G = _to_graph(G, weight, 1, np.float32)
    if dtype is not None:
        dtype = _get_float_dtype(dtype)
    elif weight in G.edge_values:
        dtype = _get_float_dtype(G.edge_values[weight].dtype)
    else:
        dtype = np.float32
    if source not in G:
        raise nx.NodeNotFound(f"Source {source} not in G")
    if G.src_indices.size == 0:
        return {source: 0}
    src_index = source if G.key_to_id is None else G.key_to_id[source]
    node_ids, distances, predecessors = plc.sssp(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(weight, 1, dtype),
        source=src_index,
        cutoff=np.inf,
        compute_predecessors=True,  # TODO: False is not yet supported
        do_expensive_check=False,
    )
    mask = distances != np.finfo(distances.dtype).max
    return G._nodearrays_to_dict(node_ids[mask], distances[mask])


@networkx_algorithm(extra_params=_dtype_param, version_added="24.04", _plc="sssp")
def single_source_bellman_ford_path(G, source, weight="weight", *, dtype=None):
    """Negative cycles are not yet supported!"""
    G = _to_graph(G, weight, 1, np.float32)
    if dtype is not None:
        dtype = _get_float_dtype(dtype)
    elif weight in G.edge_values:
        dtype = _get_float_dtype(G.edge_values[weight].dtype)
    else:
        dtype = np.float32
    if source not in G:
        raise nx.NodeNotFound(f"Source {source} not in G")
    if G.src_indices.size == 0:
        return {source: [source]}
    src_index = source if G.key_to_id is None else G.key_to_id[source]
    node_ids, distances, predecessors = plc.sssp(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(weight, 1, dtype),
        source=src_index,
        cutoff=np.inf,
        compute_predecessors=True,
        do_expensive_check=False,
    )
    mask = distances != np.finfo(distances.dtype).max
    groups = _groupby(predecessors[mask], node_ids[mask])
    if (id_to_key := G.id_to_key) is not None:
        groups = {id_to_key[k]: v for k, v in groups.items() if k >= 0}

    # Consider making utility functions for creating paths
    paths = {source: [source]}
    preds = [source]
    while preds:
        pred = preds.pop()
        pred_path = paths[pred]
        nodes = G._nodearray_to_list(groups[pred])
        for node in nodes:
            paths[node] = [*pred_path, node]
        preds.extend(nodes & groups.keys())
    return paths
