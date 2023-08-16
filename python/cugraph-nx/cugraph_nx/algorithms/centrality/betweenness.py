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
import pylibcugraph as plc

from cugraph_nx.convert import _to_graph
from cugraph_nx.utils import networkx_algorithm

__all__ = ["betweenness_centrality", "edge_betweenness_centrality"]


@networkx_algorithm
def betweenness_centrality(
    G, k=None, normalized=True, weight=None, endpoints=False, seed=None
):
    if weight is not None:
        raise NotImplementedError(
            "Weighted implementation of betweenness centrality not currently supported"
        )
    G = _to_graph(G, weight)
    node_ids, values = plc.betweenness_centrality(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        k=k,
        random_state=seed,
        normalized=normalized,
        include_endpoints=endpoints,
        do_expensive_check=False,
    )
    return G._nodearrays_to_dict(node_ids, values)


@betweenness_centrality._can_run
def _(G, k=None, normalized=True, weight=None, endpoints=False, seed=None):
    return weight is None


@networkx_algorithm
def edge_betweenness_centrality(G, k=None, normalized=True, weight=None, seed=None):
    if weight is not None:
        raise NotImplementedError(
            "Weighted implementation of betweenness centrality not currently supported"
        )
    G = _to_graph(G, weight)
    src_ids, dst_ids, values, _edge_ids = plc.edge_betweenness_centrality(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        k=k,
        random_state=seed,
        normalized=normalized,
        do_expensive_check=False,
    )
    if not G.is_directed():
        mask = src_ids <= dst_ids
        src_ids = src_ids[mask]
        dst_ids = dst_ids[mask]
        values = 2 * values[mask]
    return G._edgearrays_to_dict(src_ids, dst_ids, values)


@edge_betweenness_centrality._can_run
def _(G, k=None, normalized=True, weight=None, seed=None):
    return weight is None
