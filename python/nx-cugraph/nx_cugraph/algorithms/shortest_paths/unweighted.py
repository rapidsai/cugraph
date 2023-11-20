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

from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import index_dtype, networkx_algorithm

__all__ = ["single_source_shortest_path_length", "single_target_shortest_path_length"]


@networkx_algorithm
def single_source_shortest_path_length(G, source, cutoff=None):
    return _single_shortest_path_length(G, source, cutoff, "Source")


@networkx_algorithm
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
