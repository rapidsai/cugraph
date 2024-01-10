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
    "from_pandas_edgelist",
    "from_scipy_sparse_array",
]


@networkx_algorithm
def from_pandas_edgelist(
    df,
    source="source",
    target="target",
    edge_attr=None,
    create_using=None,
    edge_key=None,
):
    """cudf.DataFrame inputs also supported."""
    graph_class, inplace = _create_using_class(create_using)
    src_array = df[source].to_numpy()
    dst_array = df[target].to_numpy()
    # TODO: create renumbering helper function(s)
    # Renumber step 0: node keys
    nodes = np.unique(np.concatenate([src_array, dst_array]))
    N = nodes.size
    kwargs = {}
    if N > 0 and (
        nodes[0] != 0
        or nodes[N - 1] != N - 1
        or (
            nodes.dtype.kind not in {"i", "u"}
            and not (nodes == np.arange(N, dtype=np.int64)).all()
        )
    ):
        # We need to renumber indices--np.searchsorted to the rescue!
        kwargs["id_to_key"] = nodes.tolist()
        src_indices = cp.array(np.searchsorted(nodes, src_array), index_dtype)
        dst_indices = cp.array(np.searchsorted(nodes, dst_array), index_dtype)
    else:
        src_indices = cp.array(src_array)
        dst_indices = cp.array(dst_array)

    if not graph_class.is_directed():
        # Symmetrize the edges
        mask = src_indices != dst_indices
        if mask.all():
            mask = None
        src_indices, dst_indices = (
            cp.hstack(
                (src_indices, dst_indices[mask] if mask is not None else dst_indices)
            ),
            cp.hstack(
                (dst_indices, src_indices[mask] if mask is not None else src_indices)
            ),
        )

    if edge_attr is not None:
        # Additional columns requested for edge data
        if edge_attr is True:
            attr_col_headings = df.columns.difference({source, target}).to_list()
        elif isinstance(edge_attr, (list, tuple)):
            attr_col_headings = edge_attr
        else:
            attr_col_headings = [edge_attr]
        if len(attr_col_headings) == 0:
            raise nx.NetworkXError(
                "Invalid edge_attr argument: No columns found with name: "
                f"{attr_col_headings}"
            )
        try:
            edge_values = {
                key: cp.array(val.to_numpy())
                for key, val in df[attr_col_headings].items()
            }
        except (KeyError, TypeError) as exc:
            raise nx.NetworkXError(f"Invalid edge_attr argument: {edge_attr}") from exc

        if not graph_class.is_directed():
            # Symmetrize the edges
            edge_values = {
                key: cp.hstack((val, val[mask] if mask is not None else val))
                for key, val in edge_values.items()
            }
        kwargs["edge_values"] = edge_values

        if graph_class.is_multigraph() and edge_key is not None:
            try:
                edge_keys = df[edge_key].to_list()
            except (KeyError, TypeError) as exc:
                raise nx.NetworkXError(
                    f"Invalid edge_key argument: {edge_key}"
                ) from exc
            if not graph_class.is_directed():
                # Symmetrize the edges
                edge_keys = cp.hstack(
                    (edge_keys, edge_keys[mask] if mask is not None else edge_keys)
                )
            kwargs["edge_keys"] = edge_keys

    G = graph_class.from_coo(N, src_indices, dst_indices, **kwargs)
    if inplace:
        return create_using._become(G)
    return G


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
