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
from numbers import Integral

import cupy as cp
import networkx as nx
import numpy as np

from nx_cugraph.generators._utils import _create_using_class, _number_and_nodes
from nx_cugraph.utils import index_dtype, networkx_algorithm

__all__ = [
    "complete_bipartite_graph",
]


@networkx_algorithm(nodes_or_number=[0, 1], version_added="23.12")
def complete_bipartite_graph(n1, n2, create_using=None):
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    orig_n1, unused_nodes1 = n1
    orig_n2, unused_nodes2 = n2
    n1, nodes1 = _number_and_nodes(n1)
    n2, nodes2 = _number_and_nodes(n2)
    all_indices = cp.indices((n1, n2), dtype=index_dtype)
    indices0 = all_indices[0].ravel()
    indices1 = all_indices[1].ravel() + n1
    del all_indices
    src_indices = cp.hstack((indices0, indices1))
    dst_indices = cp.hstack((indices1, indices0))
    bipartite = cp.zeros(n1 + n2, np.int8)
    bipartite[n1:] = 1
    if isinstance(orig_n1, Integral) and isinstance(orig_n2, Integral):
        nodes = None
    else:
        nodes = list(range(n1)) if nodes1 is None else nodes1
        nodes.extend(range(n2) if nodes2 is None else nodes2)
        if len(set(nodes)) != len(nodes):
            raise nx.NetworkXError("Inputs n1 and n2 must contain distinct nodes")
    G = graph_class.from_coo(
        n1 + n2,
        src_indices,
        dst_indices,
        node_values={"bipartite": bipartite},
        id_to_key=nodes,
        name=f"complete_bipartite_graph({orig_n1}, {orig_n2})",
    )
    if inplace:
        return create_using._become(G)
    return G
