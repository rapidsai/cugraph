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
from __future__ import annotations

import itertools
import operator as op
from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING

import cupy as cp
import networkx as nx
import numpy as np

import nx_cugraph as nxcg

from .utils import index_dtype

if TYPE_CHECKING:  # pragma: no cover
    from nx_cugraph.typing import AttrKey, Dtype, EdgeValue, NodeValue, any_ndarray

__all__ = [
    "from_networkx",
    "to_networkx",
]

concat = itertools.chain.from_iterable
# A "required" attribute is one that all edges or nodes must have or KeyError is raised
REQUIRED = ...


def _iterate_values(graph, adj, is_dicts, func):
    # Using `dict.values` is faster and is the common case, but it doesn't always work
    if is_dicts is not False:
        it = concat(map(dict.values, adj.values()))
        if graph is not None and graph.is_multigraph():
            it = concat(map(dict.values, it))
        try:
            return func(it), True
        except TypeError:
            if is_dicts is True:
                raise
    # May not be regular dicts
    it = concat(x.values() for x in adj.values())
    if graph is not None and graph.is_multigraph():
        it = concat(x.values() for x in it)
    return func(it), False


def from_networkx(
    graph: nx.Graph,
    edge_attrs: AttrKey | dict[AttrKey, EdgeValue | None] | None = None,
    edge_dtypes: Dtype | dict[AttrKey, Dtype | None] | None = None,
    *,
    node_attrs: AttrKey | dict[AttrKey, NodeValue | None] | None = None,
    node_dtypes: Dtype | dict[AttrKey, Dtype | None] | None = None,
    preserve_all_attrs: bool = False,
    preserve_edge_attrs: bool = False,
    preserve_node_attrs: bool = False,
    preserve_graph_attrs: bool = False,
    as_directed: bool = False,
    name: str | None = None,
    graph_name: str | None = None,
) -> nxcg.Graph:
    """Convert a networkx graph to nx_cugraph graph; can convert all attributes.

    Parameters
    ----------
    G : networkx.Graph
    edge_attrs : str or dict, optional
        Dict that maps edge attributes to default values if missing in ``G``.
        If None, then no edge attributes will be converted.
        If default value is None, then missing values are handled with a mask.
        A default value of ``nxcg.convert.REQUIRED`` or ``...`` indicates that
        all edges have data for this attribute, and raise `KeyError` if not.
        For convenience, `edge_attrs` may be a single attribute with default 1;
        for example ``edge_attrs="weight"``.
    edge_dtypes : dtype or dict, optional
    node_attrs : str or dict, optional
        Dict that maps node attributes to default values if missing in ``G``.
        If None, then no node attributes will be converted.
        If default value is None, then missing values are handled with a mask.
        A default value of ``nxcg.convert.REQUIRED`` or ``...`` indicates that
        all edges have data for this attribute, and raise `KeyError` if not.
        For convenience, `node_attrs` may be a single attribute with no default;
        for example ``node_attrs="weight"``.
    node_dtypes : dtype or dict, optional
    preserve_all_attrs : bool, default False
        If True, then equivalent to setting preserve_edge_attrs, preserve_node_attrs,
        and preserve_graph_attrs to True.
    preserve_edge_attrs : bool, default False
        Whether to preserve all edge attributes.
    preserve_node_attrs : bool, default False
        Whether to preserve all node attributes.
    preserve_graph_attrs : bool, default False
        Whether to preserve all graph attributes.
    as_directed : bool, default False
        If True, then the returned graph will be directed regardless of input.
        If False, then the returned graph type is determined by input graph.
    name : str, optional
        The name of the algorithm when dispatched from networkx.
    graph_name : str, optional
        The name of the graph argument geing converted when dispatched from networkx.

    Returns
    -------
    nx_cugraph.Graph

    Notes
    -----
    For optimal performance, be as specific as possible about what is being converted:

    1. Do you need edge values? Creating a graph with just the structure is the fastest.
    2. Do you know the edge attribute(s) you need? Specify with `edge_attrs`.
    3. Do you know the default values? Specify with ``edge_attrs={weight: default}``.
    4. Do you know if all edges have values? Specify with ``edge_attrs={weight: ...}``.
    5. Do you know the dtype of attributes? Specify with `edge_dtypes`.

    Conversely, using ``preserve_edge_attrs=True`` or ``preserve_all_attrs=True`` are
    the slowest, but are also the most flexible and generic.

    See Also
    --------
    to_networkx : The opposite; convert nx_cugraph graph to networkx graph
    """
    # This uses `graph._adj` and `graph._node`, which are private attributes in NetworkX
    if not isinstance(graph, nx.Graph):
        if isinstance(graph, nx.classes.reportviews.NodeView):
            # Convert to a Graph with only nodes (no edges)
            G = nx.Graph()
            G.add_nodes_from(graph.items())
            graph = G
        else:
            raise TypeError(f"Expected networkx.Graph; got {type(graph)}")

    if preserve_all_attrs:
        preserve_edge_attrs = True
        preserve_node_attrs = True
        preserve_graph_attrs = True

    if edge_attrs is not None:
        if isinstance(edge_attrs, Mapping):
            # Copy so we don't mutate the original
            edge_attrs = dict(edge_attrs)
        else:
            edge_attrs = {edge_attrs: 1}

    if node_attrs is not None:
        if isinstance(node_attrs, Mapping):
            # Copy so we don't mutate the original
            node_attrs = dict(node_attrs)
        else:
            node_attrs = {node_attrs: None}

    if graph.__class__ in {nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph}:
        # This is a NetworkX private attribute, but is much faster to use
        adj = graph._adj
    else:
        adj = graph.adj
    if isinstance(adj, nx.classes.coreviews.FilterAdjacency):
        adj = {k: dict(v) for k, v in adj.items()}

    is_dicts = None
    N = len(adj)
    if (
        not preserve_edge_attrs
        and not edge_attrs
        # Faster than graph.number_of_edges() == 0
        or next(concat(rowdata.values() for rowdata in adj.values()), None) is None
    ):
        # Either we weren't asked to preserve edge attributes, or there are no edges
        edge_attrs = None
    elif preserve_edge_attrs:
        attr_sets, is_dicts = _iterate_values(
            graph, adj, is_dicts, lambda it: set(map(frozenset, it))
        )
        attrs = frozenset.union(*attr_sets)
        edge_attrs = dict.fromkeys(attrs, REQUIRED)
        if len(attr_sets) > 1:
            # Determine which edges have missing data
            for attr, count in Counter(concat(attr_sets)).items():
                if count != len(attr_sets):
                    edge_attrs[attr] = None
    elif None in edge_attrs.values():
        # Required edge attributes have a default of None in `edge_attrs`
        # Verify all edge attributes are present!
        required = frozenset(
            attr for attr, default in edge_attrs.items() if default is None
        )
        if len(required) == 1:
            # Fast path for the common case of a single attribute with no default
            [attr] = required
            if graph.is_multigraph():
                it = (
                    attr in edgedata
                    for rowdata in adj.values()
                    for multiedges in rowdata.values()
                    for edgedata in multiedges.values()
                )
            else:
                it = (
                    attr in edgedata
                    for rowdata in adj.values()
                    for edgedata in rowdata.values()
                )
            if next(it):
                if all(it):
                    # All edges have data
                    edge_attrs[attr] = REQUIRED
                # Else some edges have attribute (default already None)
            elif not any(it):
                # No edges have attribute
                del edge_attrs[attr]
            # Else some edges have attribute (default already None)
        else:
            attr_sets, is_dicts = _iterate_values(
                graph, adj, is_dicts, lambda it: set(map(required.intersection, it))
            )
            for attr in required - frozenset.union(*attr_sets):
                # No edges have these attributes
                del edge_attrs[attr]
            for attr in frozenset.intersection(*attr_sets):
                # All edges have these attributes
                edge_attrs[attr] = REQUIRED

    if N == 0:
        node_attrs = None
    elif preserve_node_attrs:
        attr_sets = set(map(frozenset, graph._node.values()))
        attrs = frozenset.union(*attr_sets)
        node_attrs = dict.fromkeys(attrs, REQUIRED)
        if len(attr_sets) > 1:
            # Determine which nodes have missing data
            for attr, count in Counter(concat(attr_sets)).items():
                if count != len(attr_sets):
                    node_attrs[attr] = None
    elif node_attrs and None in node_attrs.values():
        # Required node attributes have a default of None in `node_attrs`
        # Verify all node attributes are present!
        required = frozenset(
            attr for attr, default in node_attrs.items() if default is None
        )
        if len(required) == 1:
            # Fast path for the common case of a single attribute with no default
            [attr] = required
            it = (attr in nodedata for nodedata in graph._node.values())
            if next(it):
                if all(it):
                    # All nodes have data
                    node_attrs[attr] = REQUIRED
                # Else some nodes have attribute (default already None)
            elif not any(it):
                # No nodes have attribute
                del node_attrs[attr]
            # Else some nodes have attribute (default already None)
        else:
            attr_sets = set(map(required.intersection, graph._node.values()))
            for attr in required - frozenset.union(*attr_sets):
                # No nodes have these attributes
                del node_attrs[attr]
            for attr in frozenset.intersection(*attr_sets):
                # All nodes have these attributes
                node_attrs[attr] = REQUIRED

    key_to_id = dict(zip(adj, range(N)))
    dst_iter = concat(adj.values())
    try:
        no_renumber = all(k == v for k, v in key_to_id.items())
    except Exception:
        no_renumber = False
    if no_renumber:
        key_to_id = None
    else:
        dst_iter = map(key_to_id.__getitem__, dst_iter)
    if graph.is_multigraph():
        dst_indices = np.fromiter(dst_iter, index_dtype)
        num_multiedges, is_dicts = _iterate_values(
            None, adj, is_dicts, lambda it: np.fromiter(map(len, it), index_dtype)
        )
        # cp.repeat is slow to use here, so use numpy instead
        dst_indices = cp.array(np.repeat(dst_indices, num_multiedges))
        # Determine edge keys and edge ids for multigraphs
        if is_dicts:
            edge_keys = list(concat(concat(map(dict.values, adj.values()))))
            it = concat(map(dict.values, adj.values()))
        else:
            edge_keys = list(concat(concat(x.values() for x in adj.values())))
            it = concat(x.values() for x in adj.values())
        edge_indices = cp.fromiter(concat(map(range, map(len, it))), index_dtype)
        if edge_keys == edge_indices.tolist():
            edge_keys = None  # Prefer edge_indices
    else:
        dst_indices = cp.fromiter(dst_iter, index_dtype)

    edge_values = {}
    edge_masks = {}
    if edge_attrs:
        if edge_dtypes is None:
            edge_dtypes = {}
        elif not isinstance(edge_dtypes, Mapping):
            edge_dtypes = dict.fromkeys(edge_attrs, edge_dtypes)
        for edge_attr, edge_default in edge_attrs.items():
            dtype = edge_dtypes.get(edge_attr)
            if edge_default is None:
                vals = []
                append = vals.append
                if graph.is_multigraph():
                    iter_mask = (
                        append(
                            edgedata[edge_attr]
                            if (present := edge_attr in edgedata)
                            else False
                        )
                        or present
                        for rowdata in adj.values()
                        for multiedges in rowdata.values()
                        for edgedata in multiedges.values()
                    )
                else:
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
                # if vals.ndim > 1: ...
            elif edge_default is REQUIRED:
                if dtype is None:

                    def func(it, edge_attr=edge_attr):
                        return cp.array(list(map(op.itemgetter(edge_attr), it)))

                else:

                    def func(it, edge_attr=edge_attr, dtype=dtype):
                        return cp.fromiter(map(op.itemgetter(edge_attr), it), dtype)

                edge_value, is_dicts = _iterate_values(graph, adj, is_dicts, func)
                edge_values[edge_attr] = edge_value
            else:
                if graph.is_multigraph():
                    iter_values = (
                        edgedata.get(edge_attr, edge_default)
                        for rowdata in adj.values()
                        for multiedges in rowdata.values()
                        for edgedata in multiedges.values()
                    )
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
            # if vals.ndim > 1: ...

    # cp.repeat is slow to use here, so use numpy instead
    src_indices = np.repeat(
        np.arange(N, dtype=index_dtype),
        np.fromiter(map(len, adj.values()), index_dtype),
    )
    if graph.is_multigraph():
        src_indices = np.repeat(src_indices, num_multiedges)
    src_indices = cp.array(src_indices)

    node_values = {}
    node_masks = {}
    if node_attrs:
        nodes = graph._node
        if node_dtypes is None:
            node_dtypes = {}
        elif not isinstance(node_dtypes, Mapping):
            node_dtypes = dict.fromkeys(node_attrs, node_dtypes)
        for node_attr, node_default in node_attrs.items():
            # Iterate over `adj` to ensure consistent order
            dtype = node_dtypes.get(node_attr)
            if node_default is None:
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
                # Node values may be numpy or cupy arrays (useful for str, object, etc).
                # Someday we'll let the user choose np or cp, and support edge values.
                node_mask = np.fromiter(iter_mask, bool)
                node_value = np.array(vals, dtype)
                try:
                    node_value = cp.array(node_value)
                except ValueError:
                    pass
                else:
                    node_mask = cp.array(node_mask)
                node_values[node_attr] = node_value
                node_masks[node_attr] = node_mask
                # if vals.ndim > 1: ...
            else:
                if node_default is REQUIRED:
                    iter_values = (nodes[node_id][node_attr] for node_id in adj)
                else:
                    iter_values = (
                        nodes[node_id].get(node_attr, node_default) for node_id in adj
                    )
                # Node values may be numpy or cupy arrays (useful for str, object, etc).
                # Someday we'll let the user choose np or cp, and support edge values.
                if dtype is None:
                    node_value = np.array(list(iter_values))
                else:
                    node_value = np.fromiter(iter_values, dtype)
                try:
                    node_value = cp.array(node_value)
                except ValueError:
                    pass
                node_values[node_attr] = node_value
                # if vals.ndim > 1: ...
    if graph.is_multigraph():
        if graph.is_directed() or as_directed:
            klass = nxcg.MultiDiGraph
        else:
            klass = nxcg.MultiGraph
        rv = klass.from_coo(
            N,
            src_indices,
            dst_indices,
            edge_indices,
            edge_values,
            edge_masks,
            node_values,
            node_masks,
            key_to_id=key_to_id,
            edge_keys=edge_keys,
        )
    else:
        if graph.is_directed() or as_directed:
            klass = nxcg.DiGraph
        else:
            klass = nxcg.Graph
        rv = klass.from_coo(
            N,
            src_indices,
            dst_indices,
            edge_values,
            edge_masks,
            node_values,
            node_masks,
            key_to_id=key_to_id,
        )
    if preserve_graph_attrs:
        rv.graph.update(graph.graph)  # deepcopy?
    return rv


def _iter_attr_dicts(
    values: dict[AttrKey, any_ndarray[EdgeValue | NodeValue]],
    masks: dict[AttrKey, any_ndarray[bool]],
):
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


def to_networkx(G: nxcg.Graph, *, sort_edges: bool = False) -> nx.Graph:
    """Convert a nx_cugraph graph to networkx graph.

    All edge and node attributes and ``G.graph`` properties are converted.

    Parameters
    ----------
    G : nx_cugraph.Graph
    sort_edges : bool, default False
        Whether to sort the edge data of the input graph by (src, dst) indices
        before converting. This can be useful to convert to networkx graphs
        that iterate over edges consistently since edges are stored in dicts
        in the order they were added.

    Returns
    -------
    networkx.Graph

    See Also
    --------
    from_networkx : The opposite; convert networkx graph to nx_cugraph graph
    """
    rv = G.to_networkx_class()()
    id_to_key = G.id_to_key
    if sort_edges:
        G._sort_edge_indices()

    node_values = G.node_values
    node_masks = G.node_masks
    if node_values:
        node_iter = range(len(G))
        if id_to_key is not None:
            node_iter = map(id_to_key.__getitem__, node_iter)
        full_node_dicts = _iter_attr_dicts(node_values, node_masks)
        rv.add_nodes_from(zip(node_iter, full_node_dicts))
    elif id_to_key is not None:
        rv.add_nodes_from(id_to_key)
    else:
        rv.add_nodes_from(range(len(G)))

    src_indices = G.src_indices
    dst_indices = G.dst_indices
    edge_values = G.edge_values
    edge_masks = G.edge_masks
    if not G.is_directed():
        # Only add upper triangle of the adjacency matrix so we don't double-add edges
        mask = src_indices <= dst_indices
        src_indices = src_indices[mask]
        dst_indices = dst_indices[mask]
        if edge_values:
            edge_values = {k: v[mask] for k, v in edge_values.items()}
        if edge_masks:
            edge_masks = {k: v[mask] for k, v in edge_masks.items()}
    src_indices = src_iter = src_indices.tolist()
    dst_indices = dst_iter = dst_indices.tolist()
    if id_to_key is not None:
        src_iter = map(id_to_key.__getitem__, src_indices)
        dst_iter = map(id_to_key.__getitem__, dst_indices)
    if G.is_multigraph() and (G.edge_keys is not None or G.edge_indices is not None):
        if G.edge_keys is not None:
            if not G.is_directed():
                edge_keys = [k for k, m in zip(G.edge_keys, mask.tolist()) if m]
            else:
                edge_keys = G.edge_keys
        elif not G.is_directed():
            edge_keys = G.edge_indices[mask].tolist()
        else:
            edge_keys = G.edge_indices.tolist()
        if edge_values:
            full_edge_dicts = _iter_attr_dicts(edge_values, edge_masks)
            rv.add_edges_from(zip(src_iter, dst_iter, edge_keys, full_edge_dicts))
        else:
            rv.add_edges_from(zip(src_iter, dst_iter, edge_keys))
    elif edge_values:
        full_edge_dicts = _iter_attr_dicts(edge_values, edge_masks)
        rv.add_edges_from(zip(src_iter, dst_iter, full_edge_dicts))
    else:
        rv.add_edges_from(zip(src_iter, dst_iter))

    rv.graph.update(G.graph)
    return rv


def _to_graph(
    G,
    edge_attr: AttrKey | None = None,
    edge_default: EdgeValue | None = 1,
    edge_dtype: Dtype | None = None,
) -> nxcg.Graph | nxcg.DiGraph:
    """Ensure that input type is a nx_cugraph graph, and convert if necessary.

    Directed and undirected graphs are both allowed.
    This is an internal utility function and may change or be removed.
    """
    if isinstance(G, nxcg.Graph):
        return G
    if isinstance(G, nx.Graph):
        return from_networkx(
            G, {edge_attr: edge_default} if edge_attr is not None else None, edge_dtype
        )
    # TODO: handle cugraph.Graph
    raise TypeError


def _to_directed_graph(
    G,
    edge_attr: AttrKey | None = None,
    edge_default: EdgeValue | None = 1,
    edge_dtype: Dtype | None = None,
) -> nxcg.DiGraph:
    """Ensure that input type is a nx_cugraph DiGraph, and convert if necessary.

    Undirected graphs will be converted to directed.
    This is an internal utility function and may change or be removed.
    """
    if isinstance(G, nxcg.DiGraph):
        return G
    if isinstance(G, nxcg.Graph):
        return G.to_directed()
    if isinstance(G, nx.Graph):
        return from_networkx(
            G,
            {edge_attr: edge_default} if edge_attr is not None else None,
            edge_dtype,
            as_directed=True,
        )
    # TODO: handle cugraph.Graph
    raise TypeError


def _to_undirected_graph(
    G,
    edge_attr: AttrKey | None = None,
    edge_default: EdgeValue | None = 1,
    edge_dtype: Dtype | None = None,
) -> nxcg.Graph:
    """Ensure that input type is a nx_cugraph Graph, and convert if necessary.

    Only undirected graphs are allowed. Directed graphs will raise ValueError.
    This is an internal utility function and may change or be removed.
    """
    if isinstance(G, nxcg.Graph):
        if G.is_directed():
            raise ValueError("Only undirected graphs supported; got a directed graph")
        return G
    if isinstance(G, nx.Graph):
        return from_networkx(
            G, {edge_attr: edge_default} if edge_attr is not None else None, edge_dtype
        )
    # TODO: handle cugraph.Graph
    raise TypeError
