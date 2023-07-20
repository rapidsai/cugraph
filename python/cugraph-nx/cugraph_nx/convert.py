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

import collections
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

concat = itertools.chain.from_iterable


def convert_from_nx(
    graph: nx.Graph,
    edge_attrs: dict | None = None,
    node_attrs: dict | None = None,
    preserve_edge_attrs: bool = False,
    preserve_node_attrs: bool = False,
    preserve_graph_attrs: bool = False,
    name: str | None = None,
    graph_name: str | None = None,
    *,
    # Custom arguments
    is_directed: bool | None = None,
    edge_dtypes: dict | None = None,
    node_dtypes: dict | None = None,
):
    # This uses `graph._adj` and `graph._node`, which are private attributes in NetworkX
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Expected networkx.Graph; got {type(graph)}")
    if graph.is_multigraph():
        raise NotImplementedError("MultiGraph support is not yet implemented")
    has_missing_edge_data = set()
    if graph.number_of_edges() == 0:
        pass
    elif preserve_edge_attrs:
        # attrs = set().union(*concat(map(dict.values, graph._adj.values())))
        attr_sets = set(map(frozenset, concat(map(dict.values, graph._adj.values()))))
        attrs = set().union(*attr_sets)
        edge_attrs = dict.fromkeys(attrs)
        if len(attr_sets) > 1:
            counts = collections.Counter(concat(attr_sets))
            has_missing_edge_data = {
                key for key, val in counts.items() if val != len(attr_sets)
            }
    elif edge_attrs is not None and None in edge_attrs.values():
        # Required edge attributes have a default of None in `edge_attrs`
        # Verify all edge attributes are present!
        required = frozenset(
            attr for attr, default in edge_attrs.items() if default is None
        )
        attr_sets = set(
            map(required.intersection, concat(map(dict.values, graph._adj.values())))
        )
        # attr_set = set().union(*attr_sets)
        if missing := required - set().union(*attr_sets):
            # Required attributes missing completely
            missing_attrs = ", ".join(sorted(missing))
            raise TypeError(f"Missing required edge attribute: {missing_attrs}")
        if len(attr_sets) != 1:
            # Required attributes are missing _some_ data
            counts = collections.Counter(concat(attr_sets))
            bad_attrs = {key for key, val in counts.items() if val != len(attr_sets)}
            missing_attrs = ", ".join(sorted(bad_attrs))
            raise TypeError(
                f"Some edges are missing required attribute: {missing_attrs}"
            )

    has_missing_node_data = set()
    if graph.number_of_nodes() == 0:
        pass
    elif preserve_node_attrs:
        # attrs = set().union(*graph._node.values())
        attr_sets = set(map(frozenset, graph._node.values()))
        attrs = set().union(*attr_sets)
        node_attrs = dict.fromkeys(attrs)
        if len(attr_sets) > 1:
            counts = collections.Counter(concat(attr_sets))
            has_missing_node_data = {
                key for key, val in counts.items() if val != len(attr_sets)
            }
    elif node_attrs is not None and None in node_attrs.values():
        # Required node attributes have a default of None in `node_attrs`
        # Verify all node attributes are present!
        required = frozenset(
            attr for attr, default in node_attrs.items() if default is None
        )
        attr_sets = set(map(required.intersection, graph._node.values()))
        if missing := required - set().union(*attr_sets):
            # Required attributes missing completely
            missing_attrs = ", ".join(sorted(missing))
            raise TypeError(f"Missing required node attribute: {missing_attrs}")
        if len(attr_sets) != 1:
            # Required attributes are missing _some_ data
            counts = collections.Counter(concat(attr_sets))
            bad_attrs = {key for key, val in counts.items() if val != len(attr_sets)}
            missing_attrs = ", ".join(sorted(bad_attrs))
            raise TypeError(
                f"Some nodes are missing required attribute: {missing_attrs}"
            )

    get_edge_values = edge_attrs is not None
    adj = graph._adj  # This is a NetworkX private attribute, but is much faster to use
    if get_edge_values and isinstance(adj, nx.classes.coreviews.FilterAdjacency):
        adj = {k: dict(v) for k, v in adj.items()}
    N = len(adj)
    key_to_id = dict(zip(adj, range(N)))
    do_remap = not all(k == v for k, v in key_to_id.items())
    col_iter = itertools.chain.from_iterable(adj.values())
    if do_remap:
        col_iter = map(key_to_id.__getitem__, col_iter)
    else:
        key_to_id = None
    # TODO: do col_indices need to be sorted in each row (if we use indptr as CSR)?
    col_indices = cp.fromiter(col_iter, np.int32)

    edge_values = {}
    edge_masks = {}
    if get_edge_values:
        if edge_dtypes is None:
            edge_dtypes = {}
        for edge_attr, edge_default in edge_attrs.items():
            dtype = edge_dtypes.get(edge_attr)
            if edge_default is None and edge_attr in has_missing_edge_data:
                vals = []
                append = vals.append
                iter_mask = (
                    append(
                        edgedata[edge_attr]
                        if (present := edge_attr in edgedata)
                        else False
                    )
                    or present
                    for rowdata in adj.values()
                    for edgedata in rowdata.values()
                )
                edge_masks[edge_attr] = cp.fromiter(iter_mask, bool)
                edge_values[edge_attr] = cp.array(vals, dtype)
            else:
                iter_values = (
                    edgedata.get(edge_attr, edge_default)
                    for rowdata in adj.values()
                    for edgedata in rowdata.values()
                )
                if dtype is None:
                    edge_values[edge_attr] = cp.array(list(iter_values))
                else:
                    edge_values[edge_attr] = cp.fromiter(iter_values, dtype)
                if edge_default is None:
                    edge_masks[edge_attr] = cp.zeros(col_indices.size, bool)

    # TODO: should we use indptr for CSR? Should we only use COO?
    indptr = cp.cumsum(
        cp.fromiter(itertools.chain([0], map(len, adj.values())), np.int32),
        dtype=np.int32,
    )
    row_indices = cp.repeat(cp.arange(N, dtype=np.int32), list(map(len, adj.values())))

    get_node_values = node_attrs is not None
    node_values = {}
    node_masks = {}
    nodes = graph._node
    if get_node_values:
        if node_dtypes is None:
            node_dtypes = {}
        for node_attr, node_default in node_attrs.items():
            # Iterate over `adj` to ensure consistent order
            dtype = node_dtypes.get(node_attr)
            if node_default is None and node_attr in has_missing_node_data:
                vals = []
                append = vals.append
                iter_mask = (
                    append(
                        nodedata[node_attr]
                        if (present := node_attr in (nodedata := nodes[node_id]))
                        else False
                    )
                    or present
                    for node_id in adj
                )
                node_masks[node_attr] = cp.fromiter(iter_mask, bool)
                node_values[node_attr] = cp.array(vals, dtype)
            else:
                iter_values = (
                    nodes[node_id].get(node_attr, node_default) for node_id in adj
                )
                if dtype is None:
                    node_values[node_attr] = cp.array(list(iter_values))
                else:
                    node_values[node_attr] = cp.fromiter(iter_values, dtype)
                if node_default is None:
                    node_masks[node_attr] = cp.zeros(col_indices.size, bool)

    if graph.is_directed() or is_directed:
        klass = cnx.DiGraph
    else:
        klass = cnx.Graph
    rv = klass(
        indptr,
        row_indices,
        col_indices,
        edge_values,
        edge_masks,
        node_values,
        node_masks,
        key_to_id=key_to_id,
    )
    if preserve_graph_attrs:
        rv.graph.update(graph.graph)  # deepcopy?
    return rv


def from_networkx(
    G: nx.Graph,
    edge_attr=None,
    edge_default=1.0,
    *,
    is_directed: bool | None = None,
    dtype=None,
) -> cnx.Graph:
    if edge_attr is not None:
        edge_attrs = {edge_attr: edge_default}
        edge_dtypes = {edge_attr: dtype}
    else:
        edge_attrs = edge_dtypes = None
    return convert_from_nx(
        G, edge_attrs=edge_attrs, is_directed=is_directed, edge_dtypes=edge_dtypes
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
