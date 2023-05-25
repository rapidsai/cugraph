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

from cugraph_nx.convert import to_directed_graph
from cugraph_nx.utils.decorators import not_implemented_for

__all__ = ["is_strongly_connected"]


@not_implemented_for("undirected")
def is_strongly_connected(G):
    G = to_directed_graph(G)
    N = len(G)
    if N == 0:
        raise nx.NetworkXPointlessConcept(
            "Connectivity is undefined for the null graph."
        )
    rv = cp.empty(N, np.int32)
    plc.strongly_connected_components(
        G.indptr, G.col_indices, None, N, G.col_indices.size, rv
    )
    return bool((rv[0] == rv).all())
