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

from cugraph_nx.utils import networkx_api

__all__ = ["betweenness_centrality"]


@networkx_api
def betweenness_centrality(
    G, k=None, normalized=True, weight=None, endpoints=False, seed=None
):
    if weight is not None:
        raise NotImplementedError(
            "Weighted implementation of betweenness centrality not currently supported"
        )
    node_ids, values = plc.betweenness_centrality(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        k=k,
        random_state=seed,
        normalized=normalized,
        include_endpoints=endpoints,
        do_expensive_check=False,
    )
    return G._nodeids_to_dict(node_ids, values)
