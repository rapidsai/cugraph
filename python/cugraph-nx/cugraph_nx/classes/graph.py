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

from copy import deepcopy
from typing import TYPE_CHECKING, ClassVar

import cupy as cp
import networkx as nx
import numpy as np
import pylibcugraph as plc

import cugraph_nx as cnx

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cugraph_nx.typing import AttrKey, EdgeValue, NodeKey

__all__ = ["Graph"]

networkx_api = cnx.utils.networkx_api(api=nx.Graph)


class Graph:
    row_indices: cp.ndarray
    col_indices: cp.ndarray
    edge_values: dict[AttrKey, cp.ndarray]
    edge_masks: dict[AttrKey, cp.ndarray[bool]]
    node_values: dict[AttrKey, cp.ndarray]
    node_masks: dict[AttrKey, cp.ndarray[bool]]
    key_to_id: dict[NodeKey, int] | None
    _id_to_key: dict[int, NodeKey] | None
    _N: int
    graph: dict

    __networkx_plugin__: ClassVar[str] = "cugraph"

    graph_attr_dict_factory: ClassVar[type] = dict

    ####################
    # Creation methods #
    ####################

    @classmethod
    def from_coo(
        cls,
        N: int,
        row_indices: cp.ndarray,
        col_indices: cp.ndarray,
        edge_values: dict[AttrKey, cp.ndarray] | None = None,
        edge_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        node_values: dict[AttrKey, cp.ndarray] | None = None,
        node_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        *,
        key_to_id: dict[NodeKey, int] | None = None,
        id_to_key: dict[int, NodeKey] | None = None,
        **attr,
    ) -> Graph:
        self = object.__new__(cls)
        self.row_indices = row_indices
        self.col_indices = col_indices
        self.edge_values = {} if edge_values is None else edge_values
        self.edge_masks = {} if edge_masks is None else edge_masks
        self.node_values = {} if node_values is None else node_values
        self.node_masks = {} if node_masks is None else node_masks
        self.key_to_id = key_to_id
        self._id_to_key = id_to_key
        self._N = N
        self.graph = self.graph_attr_dict_factory()
        self.graph.update(attr)
        size = self.row_indices.size
        # Easy and fast sanity checks
        if size != self.col_indices.size:
            raise ValueError
        for attr in ["edge_values", "edge_masks"]:
            if datadict := getattr(self, attr):
                for key, val in datadict.items():
                    if val.shape[0] != size:
                        raise ValueError(key)
        for attr in ["node_values", "node_masks"]:
            if datadict := getattr(self, attr):
                for key, val in datadict.items():
                    if val.shape[0] != N:
                        raise ValueError(key)
        if self.key_to_id is not None and len(self.key_to_id) != N:
            raise ValueError
        if self._id_to_key is not None and len(self._id_to_key) != N:
            raise ValueError
        return self

    @classmethod
    def from_csr(
        cls,
        indptr: cp.ndarray,
        col_indices: cp.ndarray,
        edge_values: dict[AttrKey, cp.ndarray] | None = None,
        edge_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        node_values: dict[AttrKey, cp.ndarray] | None = None,
        node_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        *,
        key_to_id: dict[NodeKey, int] | None = None,
        id_to_key: dict[int, NodeKey] | None = None,
        **attr,
    ) -> Graph:
        N = indptr.size - 1
        row_indices = cp.repeat(cp.arange(N, dtype=np.int32), cp.diff(indptr).tolist())
        return cls.from_coo(
            N,
            row_indices,
            col_indices,
            edge_values,
            edge_masks,
            node_values,
            node_masks,
            key_to_id=key_to_id,
            id_to_key=id_to_key,
            **attr,
        )

    @classmethod
    def from_csc(
        cls,
        indptr: cp.ndarray,
        row_indices: cp.ndarray,
        edge_values: dict[AttrKey, cp.ndarray] | None = None,
        edge_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        node_values: dict[AttrKey, cp.ndarray] | None = None,
        node_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        *,
        key_to_id: dict[NodeKey, int] | None = None,
        id_to_key: dict[int, NodeKey] | None = None,
        **attr,
    ) -> Graph:
        N = indptr.size - 1
        col_indices = cp.repeat(cp.arange(N, dtype=np.int32), cp.diff(indptr).tolist())
        return cls.from_coo(
            N,
            row_indices,
            col_indices,
            edge_values,
            edge_masks,
            node_values,
            node_masks,
            key_to_id=key_to_id,
            id_to_key=id_to_key,
            **attr,
        )

    @classmethod
    def from_dcsr(
        cls,
        N: int,
        compressed_rows: cp.ndarray,
        indptr: cp.ndarray,
        col_indices: cp.ndarray,
        edge_values: dict[AttrKey, cp.ndarray] | None = None,
        edge_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        node_values: dict[AttrKey, cp.ndarray] | None = None,
        node_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        *,
        key_to_id: dict[NodeKey, int] | None = None,
        id_to_key: dict[int, NodeKey] | None = None,
        **attr,
    ) -> Graph:
        row_indices = cp.repeat(compressed_rows, cp.diff(indptr).tolist())
        return cls.from_coo(
            N,
            row_indices,
            col_indices,
            edge_values,
            edge_masks,
            node_values,
            node_masks,
            key_to_id=key_to_id,
            id_to_key=id_to_key,
            **attr,
        )

    @classmethod
    def from_dcsc(
        cls,
        N: int,
        compressed_cols: cp.ndarray,
        indptr: cp.ndarray,
        row_indices: cp.ndarray,
        edge_values: dict[AttrKey, cp.ndarray] | None = None,
        edge_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        node_values: dict[AttrKey, cp.ndarray] | None = None,
        node_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        *,
        key_to_id: dict[NodeKey, int] | None = None,
        id_to_key: dict[int, NodeKey] | None = None,
        **attr,
    ) -> Graph:
        col_indices = cp.repeat(compressed_cols, cp.diff(indptr).tolist())
        return cls.from_coo(
            N,
            row_indices,
            col_indices,
            edge_values,
            edge_masks,
            node_values,
            node_masks,
            key_to_id=key_to_id,
            id_to_key=id_to_key,
            **attr,
        )

    def __new__(cls, incoming_graph_data=None, **attr) -> Graph:
        if incoming_graph_data is None:
            self = cls.from_coo(0, cp.empty(0, np.int32), cp.empty(0, np.int32))
        elif incoming_graph_data.__class__ is self.__class__:
            self = incoming_graph_data.copy()
        elif incoming_graph_data.__class__ is self.to_networkx_class():
            self = cnx.from_networkx(incoming_graph_data, preserve_all_attrs=True)
        else:
            raise NotImplementedError
        self.graph.update(attr)
        return self

    #################
    # Class methods #
    #################

    @classmethod
    @networkx_api
    def is_directed(cls) -> bool:
        return False

    @classmethod
    @networkx_api
    def is_multigraph(cls) -> bool:
        return False

    @classmethod
    @networkx_api
    def to_directed_class(cls):
        return cnx.DiGraph

    @classmethod
    def to_networkx_class(cls):
        return nx.Graph

    @classmethod
    @networkx_api
    def to_undirected_class(cls):
        return Graph

    ##############
    # Properties #
    ##############

    @property
    def id_to_key(self) -> dict[int, NodeKey] | None:
        if self.key_to_id is None:
            return None
        if self._id_to_key is None:
            self._id_to_key = {val: key for key, val in self.key_to_id.items()}
        return self._id_to_key

    name = nx.Graph.name

    ##################
    # Dunder methods #
    ##################

    @networkx_api
    def __contains__(self, n: NodeKey) -> bool:
        if self.key_to_id is not None:
            container = self.key_to_id
        else:
            container = range(self._N)
        try:
            return n in container
        except TypeError:
            return False

    @networkx_api
    def __iter__(self) -> Iterator[NodeKey]:
        if self.key_to_id is not None:
            return iter(self.key_to_id)
        return iter(range(self._N))

    @networkx_api
    def __len__(self) -> int:
        return self._N

    __str__ = nx.Graph.__str__

    ##########################
    # NetworkX graph methods #
    ##########################

    @networkx_api
    def clear(self) -> None:
        self.edge_values.clear()
        self.edge_masks.clear()
        self.node_values.clear()
        self.node_masks.clear()
        self.graph.clear()
        self.row_indices = cp.empty(0, self.row_indices.dtype)
        self.col_indices = cp.empty(0, self.col_indices.dtype)
        self._N = 0
        self.key_to_id = None
        self._id_to_key = None

    @networkx_api
    def clear_edges(self) -> None:
        self.edge_values.clear()
        self.edge_masks.clear()
        self.row_indices = cp.empty(0, self.row_indices.dtype)
        self.col_indices = cp.empty(0, self.col_indices.dtype)

    @networkx_api
    def copy(self, as_view: bool = False) -> Graph:
        # Does shallow copy in networkx
        return self._copy(as_view, self.__class__)

    @networkx_api
    def get_edge_data(
        self, u: NodeKey, v: NodeKey, default: EdgeValue | None = None
    ) -> dict[AttrKey, EdgeValue]:
        if self.key_to_id is not None:
            try:
                u = self.key_to_id[u]
                v = self.key_to_id[v]
            except KeyError:
                return default
        index = cp.nonzero((self.row_indices == u) & (self.col_indices == v))[0]
        if index.size == 0:
            return default
        [index] = index.tolist()
        if not self.edge_values:
            return {}
        return {
            key: val[index].tolist()
            for key, val in self.edge_values.items()
            if key not in self.edge_masks or self.edge_masks[key][index]
        }

    @networkx_api
    def has_edge(self, u: NodeKey, v: NodeKey) -> bool:
        if self.key_to_id is not None:
            try:
                u = self.key_to_id[u]
                v = self.key_to_id[v]
            except KeyError:
                return False
        return bool(((self.row_indices == u) & (self.col_indices == v)).any())

    @networkx_api
    def has_node(self, n: NodeKey) -> bool:
        return n in self

    @networkx_api
    def nbunch_iter(self, nbunch=None) -> Iterator[NodeKey]:
        if nbunch is None:
            return iter(self)
        if nbunch in self:
            return iter([nbunch])
        return (node for node in nbunch if node in self)

    @networkx_api
    def number_of_edges(
        self, u: NodeKey | None = None, v: NodeKey | None = None
    ) -> int:
        if u is not None or v is not None:
            raise NotImplementedError
        return self.size()

    @networkx_api
    def number_of_nodes(self) -> int:
        return self._N

    @networkx_api
    def order(self) -> int:
        return self._N

    @networkx_api
    def size(self, weight: AttrKey | None = None) -> int:
        if weight is not None:
            raise NotImplementedError
        # If no self-edges, then `self.row_indices.size // 2`
        return int((self.row_indices <= self.col_indices).sum())

    @networkx_api
    def to_directed(self, as_view: bool = False) -> cnx.DiGraph:
        return self._copy(as_view, self.to_directed_class())

    @networkx_api
    def to_undirected(self, as_view: bool = False) -> Graph:
        # Does deep copy in networkx
        return self.copy(as_view)

    # Not implemented...
    # adj, adjacency, add_edge, add_edges_from, add_node,
    # add_nodes_from, add_weighted_edges_from, degree,
    # edge_subgraph, edges, neighbors, nodes, remove_edge,
    # remove_edges_from, remove_node, remove_nodes_from, subgraph, update

    ###################
    # Private methods #
    ###################

    def _copy(self, as_view, cls, reverse=False):
        indptr = self.indptr
        row_indices = self.row_indices
        col_indices = self.col_indices
        edge_values = self.edge_values
        edge_masks = self.edge_masks
        node_values = self.node_values
        node_masks = self.node_masks
        key_to_id = self.key_to_id
        id_to_key = None if key_to_id is None else self._id_to_key
        if not as_view:
            indptr = indptr.copy()
            row_indices = row_indices.copy()
            col_indices = col_indices.copy()
            edge_values = {key: val.copy() for key, val in edge_values.items()}
            edge_masks = {key: val.copy() for key, val in edge_masks.items()}
            node_values = {key: val.copy() for key, val in node_values.items()}
            node_masks = {key: val.copy() for key, val in node_masks.items()}
            if key_to_id is not None:
                key_to_id = key_to_id.copy()
                if id_to_key is not None:
                    id_to_key = id_to_key.copy()
        if reverse:
            row_indices, col_indices = col_indices, row_indices
        rv = cls.from_coo(
            indptr,
            row_indices,
            col_indices,
            edge_values,
            edge_masks,
            node_values,
            node_masks,
            key_to_id=key_to_id,
            id_to_key=id_to_key,
        )
        if as_view:
            rv.graph = self.graph
        else:
            rv.graph.update(deepcopy(self.graph))
        return rv

    def _get_plc_graph(
        self, edge_attr: AttrKey = None, *, store_transposed: bool = False
    ):
        if edge_attr is None:
            edge_array = None
        elif edge_attr not in self.edge_values:
            raise KeyError("Graph has no edge attribute {edge_attr!r}")
        elif edge_attr not in self.edge_masks:
            edge_array = self.edge_values[edge_attr]
        elif not self.edge_masks[edge_attr].all():
            raise NotImplementedError("Missing edge attributes is not yet implemented")
        else:
            # Mask is all True; don't need anymore
            del self.edge_masks[edge_attr]
            edge_array = self.edge_values[edge_attr]
        # Should we cache PLC graph?
        return plc.SGGraph(
            resource_handle=plc.ResourceHandle(),
            graph_properties=plc.GraphProperties(
                is_multigraph=self.is_multigraph(),
                is_symmetric=not self.is_directed(),
            ),
            src_or_offset_array=self.row_indices,
            dst_or_index_array=self.col_indices,
            weight_array=edge_array,
            store_transposed=store_transposed,
            renumber=False,
            do_expensive_check=False,
        )

    def _nodes_to_dict(self, node_ids: cp.ndarray, values: cp.ndarray):
        it = zip(node_ids.tolist(), values.tolist())
        if (id_to_key := self.id_to_key) is not None:
            return {id_to_key[key]: val for key, val in it}
        return dict(it)

    def _edges_to_dict(
        self, src_ids: cp.ndarray, dst_ids: cp.ndarray, values: cp.ndarray
    ):
        it = zip(zip(src_ids.tolist(), dst_ids.tolist()), values.tolist())
        if (id_to_key := self.id_to_key) is not None:
            return {
                (id_to_key[src_id], id_to_key[dst_id]): val
                for (src_id, dst_id), val in it
            }
        return dict(it)
