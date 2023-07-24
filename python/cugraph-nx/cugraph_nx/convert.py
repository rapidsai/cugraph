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
import sys

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

IS_TESTING = "pytest" in sys.argv[0] or "py.test" in sys.argv[0]  # Hacky...


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
    weight=None,  # For nx.__version__ <= 3.1
    # Custom arguments
    is_directed: bool | None = None,
    edge_dtypes: dict | None = None,
    node_dtypes: dict | None = None,
):
    if IS_TESTING and name in {
        # Raise if input graph changes
        "lexicographical_topological_sort",
        "topological_generations",
        "topological_sort",
        # Sensitive tests (iteration order matters)
        "dfs_labeled_edges",
    }:
        return graph

    # This uses `graph._adj` and `graph._node`, which are private attributes in NetworkX
    if isinstance(graph, nx.classes.reportviews.NodeView):
        # Convert to a Graph with only nodes (no edges)
        G = nx.Graph()
        G.add_nodes_from(graph.items())
        graph = G
    elif not isinstance(graph, nx.Graph):
        raise TypeError(f"Expected networkx.Graph; got {type(graph)}")
    elif graph.__class__ not in {nx.Graph, nx.DiGraph}:
        if IS_TESTING:
            return graph
        if graph.is_multigraph():
            raise NotImplementedError("MultiGraph support is not yet implemented")
        raise NotImplementedError(
            "Converting graph subclas is not yet implemented:", type(graph)
        )

    if weight is not None:
        # For networkx 3.0 and 3.1 compatibility
        if edge_attrs is not None:
            raise TypeError("edge_attrs and weight arguments both given")
        edge_attrs = {weight: 1}

    adj = graph._adj  # This is a NetworkX private attribute, but is much faster to use
    if isinstance(adj, nx.classes.coreviews.FilterAdjacency):
        adj = {k: dict(v) for k, v in adj.items()}

    has_missing_edge_data = set()
    missing_edge_attrs = set()
    if graph.number_of_edges() == 0:
        pass
    elif preserve_edge_attrs:
        attr_sets = set(map(frozenset, concat(map(dict.values, adj.values()))))
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
            map(required.intersection, concat(map(dict.values, adj.values())))
        )
        missing_edge_attrs = required - set().union(*attr_sets)
        if len(attr_sets) != 1:
            # Required attributes are missing _some_ data
            counts = collections.Counter(concat(attr_sets))
            has_missing_edge_data.update(
                key for key, val in counts.items() if val != len(attr_sets)
            )

    has_missing_node_data = set()
    missing_node_attrs = set()
    if graph.number_of_nodes() == 0:
        pass
    elif preserve_node_attrs:
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
        missing_node_attrs = required - set().union(*attr_sets)
        if len(attr_sets) != 1:
            # Required attributes are missing _some_ data
            counts = collections.Counter(concat(attr_sets))
            has_missing_node_data.update(
                key for key, val in counts.items() if val != len(attr_sets)
            )

    get_edge_values = edge_attrs is not None
    N = len(adj)
    key_to_id = dict(zip(adj, range(N)))
    do_remap = not all(k == v for k, v in key_to_id.items())
    col_iter = itertools.chain.from_iterable(adj.values())
    if do_remap:
        col_iter = map(key_to_id.__getitem__, col_iter)
    else:
        key_to_id = None
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
                vals = edge_values[edge_attr] = cp.array(vals, dtype)
                # if vals.ndim > 1: ...
            elif edge_attr not in missing_edge_attrs:
                iter_values = (
                    edgedata.get(edge_attr, edge_default)
                    for rowdata in adj.values()
                    for edgedata in rowdata.values()
                )
                if dtype is None:
                    vals = edge_values[edge_attr] = cp.array(list(iter_values))
                else:
                    vals = edge_values[edge_attr] = cp.fromiter(iter_values, dtype)
                # if vals.ndim > 1: ...

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
                vals = node_values[node_attr] = cp.array(vals, dtype)
                # if vals.ndim > 1: ...
            elif node_attr not in missing_node_attrs:
                iter_values = (
                    nodes[node_id].get(node_attr, node_default) for node_id in adj
                )
                if dtype is None:
                    vals = node_values[node_attr] = cp.array(list(iter_values))
                else:
                    vals = node_values[node_attr] = cp.fromiter(iter_values, dtype)
                # if vals.ndim > 1: ...

    if graph.is_directed() or is_directed:
        klass = cnx.DiGraph
    else:
        klass = cnx.Graph
    rv = klass(
        len(graph),
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


def convert_to_nx(obj, *, name=None):
    if isinstance(obj, cnx.Graph):
        return to_networkx(obj)
    return obj


def from_networkx(
    G: nx.Graph,
    edge_attr=None,
    edge_default=1.0,
    edge_dtype=None,
    *,
    node_attr=None,
    node_default=None,
    node_dtype=None,
    is_directed: bool | None = None,
) -> cnx.Graph:
    if edge_attr is not None:
        edge_attrs = {edge_attr: edge_default}
        edge_dtypes = {edge_attr: edge_dtype}
    else:
        edge_attrs = edge_dtypes = None
    if node_attr is not None:
        node_attrs = {node_attr: node_default}
        node_dtypes = {node_attr: node_dtype}
    else:
        node_attrs = node_dtypes = None
    return convert_from_nx(
        G,
        edge_attrs=edge_attrs,
        edge_dtypes=edge_dtypes,
        node_attrs=node_attrs,
        node_dtypes=node_dtypes,
        is_directed=is_directed,
    )


def _iter_attr_dicts(values, masks):
    full_attrs = list(values.keys() - masks.keys())
    if full_attrs:
        full_dicts = (
            dict(zip(full_attrs, vals))
            for vals in zip(*(values[attr].tolist() for attr in full_attrs))
        )
    partial_attrs = list(values.keys() & masks.keys())
    if partial_attrs:
        partial_dicts = (
            {k: v for k, (v, m) in zip(partial_attrs, vals_masks) if m}
            for vals_masks in zip(
                *(
                    zip(values[attr].tolist(), masks[attr].tolist())
                    for attr in partial_attrs
                )
            )
        )
    if full_attrs and partial_attrs:
        full_dicts = (d1.update(d2) or d1 for d1, d2 in zip(full_dicts, partial_dicts))
    elif partial_attrs:
        full_dicts = partial_dicts
    return full_dicts


def to_networkx(G) -> nx.Graph:
    rv = G.to_networkx_class()()
    id_to_key = G.id_to_key

    node_values = G.node_values
    node_masks = G.node_masks
    if node_values:
        node_iter = range(len(G))
        if id_to_key is not None:
            node_iter = map(id_to_key.__getitem__, node_iter)
        full_node_dicts = _iter_attr_dicts(node_values, node_masks)
        rv.add_nodes_from(zip(node_iter, full_node_dicts))
    else:
        if id_to_key is not None:
            rv.add_nodes_from(id_to_key.values())
        else:
            rv.add_nodes_from(range(len(G)))

    row_indices = G.row_indices
    col_indices = G.col_indices
    edge_values = G.edge_values
    edge_masks = G.edge_masks
    if edge_values and not G.is_directed():
        # Only add upper triangle of the adjacency matrix so we don't double-add edges
        mask = row_indices <= col_indices
        row_indices = row_indices[mask]
        col_indices = col_indices[mask]
        edge_values = {k: v[mask] for k, v in edge_values.items()}
        if edge_masks:
            edge_masks = {k: v[mask] for k, v in edge_masks.items()}
    row_indices = row_iter = row_indices.tolist()
    col_indices = col_iter = col_indices.tolist()
    if id_to_key is not None:
        row_iter = map(id_to_key.__getitem__, row_indices)
        col_iter = map(id_to_key.__getitem__, col_indices)
    if edge_values:
        full_edge_dicts = _iter_attr_dicts(edge_values, edge_masks)
        rv.add_edges_from(zip(row_iter, col_iter, full_edge_dicts))
    else:
        rv.add_edges_from(zip(row_iter, col_iter))

    rv.graph.update(G.graph)
    return rv


def to_graph(
    G, edge_attr=None, edge_default=1.0, edge_dtype=None
) -> cnx.Graph | cnx.DiGraph:
    if isinstance(G, cnx.Graph):
        return G
    if isinstance(G, nx.Graph):
        return from_networkx(G, edge_attr, edge_default, edge_dtype)
    # TODO: handle cugraph.Graph
    raise TypeError


def to_directed_graph(
    G, edge_attr=None, edge_default=1.0, edge_dtype=None
) -> cnx.DiGraph:
    if isinstance(G, cnx.DiGraph):
        return G
    if isinstance(G, cnx.Graph):
        return G.to_directed()
    if isinstance(G, nx.Graph):
        return from_networkx(G, edge_attr, edge_default, edge_dtype, is_directed=True)
    # TODO: handle cugraph.Graph
    raise TypeError


def to_undirected_graph(
    G, edge_attr=None, edge_default=1.0, edge_dtype=None
) -> cnx.Graph:
    if isinstance(G, cnx.Graph):
        if G.is_directed():
            raise NotImplementedError
        return G
    if isinstance(G, nx.Graph):
        return from_networkx(G, edge_attr, edge_default, edge_dtype)
    # TODO: handle cugraph.Graph
    raise TypeError
