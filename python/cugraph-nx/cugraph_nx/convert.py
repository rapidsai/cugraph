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
from __future__ import annotations

import itertools

import cupy as cp
import networkx as nx
import numpy as np

import cugraph_nx as cnx

__all__ = [
    "from_networkx",
    "to_networkx",
    "to_graph",
    "to_directed_graph",
    "to_undirected_graph",
]


def from_networkx(
    G, edge_attr=None, edge_default=1.0, *, is_directed=None, dtype=None
) -> cnx.Graph:
    if not isinstance(G, nx.Graph):
        raise TypeError(f"Expected networkx.Graph; got {type(G)}")
    if G.is_multigraph():
        raise NotImplementedError("MultiGraph support is not yet implemented")
    if isinstance(edge_attr, (list, dict, set)):
        raise NotImplementedError(
            "Graph with multiple attributes is not yet supported; "
            f"bad edge_attr: {edge_attr}"
        )
    get_values = edge_attr is not None
    adj = G._adj  # This is a NetworkX private attribute, but is much faster to use
    if get_values and isinstance(adj, nx.classes.coreviews.FilterAdjacency):
        adj = {k: dict(v) for k, v in adj.items()}
    N = len(adj)
    key_to_id = dict(zip(adj, range(N)))
    do_remap = not all(k == v for k, v in key_to_id.items())
    col_iter = itertools.chain.from_iterable(adj.values())
    if do_remap:
        col_iter = map(key_to_id.__getitem__, col_iter)
    else:
        key_to_id = None
    # TODO: do col_indices need to be sorted in each row?
    col_indices = cp.fromiter(col_iter, np.int32)
    iter_values = (
        edgedata.get(edge_attr, edge_default)
        for rowdata in adj.values()
        for edgedata in rowdata.values()
    )
    if not get_values:
        values = None
    elif dtype is None:
        values = cp.array(list(iter_values))
    else:
        values = cp.fromiter(iter_values, dtype)
    indptr = cp.cumsum(
        cp.fromiter(itertools.chain([0], map(len, adj.values())), np.int32),
        dtype=np.int32,
    )
    row_indices = cp.repeat(cp.arange(N, dtype=np.int32), list(map(len, adj.values())))
    if G.is_directed() or is_directed:
        klass = cnx.DiGraph
    else:
        klass = cnx.Graph
    return klass(
        indptr,
        row_indices,
        col_indices,
        values,
        key_to_id=key_to_id,
    )


def to_networkx(G) -> nx.Graph:
    # There are many paths to convert to NetworkX:
    # pandas, scipy.sparse, nx.from_dict_of_lists, etc.
    # which means pandas/scipy can be optional dependencies even here.
    import pandas as pd

    d = {"source": G.row_indices.get(), "target": G.col_indices.get()}
    if G.edge_values is not None:
        d["weight"] = G.edge_values.get()
        edge_attr = "weight"
    else:
        edge_attr = None
    df = pd.DataFrame(d)
    if G.key_to_id is not None:
        df["source"] = df["source"].map(G.id_to_key)
        df["target"] = df["target"].map(G.id_to_key)
    return nx.from_pandas_edgelist(
        df,
        edge_attr=edge_attr,
        create_using=nx.DiGraph if G.is_directed() else nx.Graph,
    )


def to_graph(
    G, edge_attr=None, edge_default=1.0, *, dtype=None
) -> cnx.Graph | cnx.DiGraph:
    if isinstance(G, cnx.Graph):
        return G
    if isinstance(G, nx.Graph):
        return from_networkx(G, edge_attr, edge_default, dtype=dtype)
    # TODO: handle cugraph.Graph
    raise TypeError


def to_directed_graph(
    G, edge_attr=None, edge_default=1.0, *, dtype=None
) -> cnx.DiGraph:
    if isinstance(G, cnx.DiGraph):
        return G
    if isinstance(G, cnx.Graph):
        return G.to_directed()
    if isinstance(G, nx.Graph):
        return from_networkx(G, edge_attr, edge_default, is_directed=True, dtype=dtype)
    # TODO: handle cugraph.Graph
    raise TypeError


def to_undirected_graph(
    G, edge_attr=None, edge_default=1.0, *, dtype=None
) -> cnx.Graph:
    if isinstance(G, cnx.Graph):
        if G.is_directed():
            raise NotImplementedError
        return G
    if isinstance(G, nx.Graph):
        return from_networkx(G, edge_attr, edge_default, dtype=dtype)
    # TODO: handle cugraph.Graph
    raise TypeError
