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

from .generators._utils import _create_using_class
from .utils import index_dtype, networkx_algorithm

__all__ = [
    "from_scipy_sparse_array",
]


@networkx_algorithm
def from_scipy_sparse_array(
    A, parallel_edges=False, create_using=None, edge_attribute="weight"
):
    graph_class, inplace = _create_using_class(create_using)
    m, n = A.shape
    if m != n:
        raise nx.NetworkXError(f"Adjacency matrix not square: nx,ny={A.shape}")
    if A.format != "coo":
        A = A.tocoo()
    if A.dtype.kind in {"i", "u"} and graph_class.is_multigraph() and parallel_edges:
        src_indices = cp.array(np.repeat(A.row, A.data), index_dtype)
        dst_indices = cp.array(np.repeat(A.col, A.data), index_dtype)
        weight = cp.empty(src_indices.size, A.data.dtype)
        weight[:] = 1
    else:
        src_indices = cp.array(A.row, index_dtype)
        dst_indices = cp.array(A.col, index_dtype)
        weight = cp.array(A.data)
    G = graph_class.from_coo(
        n, src_indices, dst_indices, edge_values={"weight": weight}
    )
    if inplace:
        return create_using._become(G)
    return G
