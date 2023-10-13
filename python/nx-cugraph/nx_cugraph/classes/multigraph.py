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

import nx_cugraph as nxcg

from .graph import Graph

if TYPE_CHECKING:
    from nx_cugraph.typing import (
        AttrKey,
        EdgeKey,
        EdgeValue,
        IndexValue,
        NodeKey,
        NodeValue,
    )
__all__ = ["MultiGraph"]

networkx_api = nxcg.utils.decorators.networkx_class(nx.MultiGraph)


class MultiGraph(Graph):
    # networkx properties
    edge_key_dict_factory: ClassVar[type] = dict

    # Not networkx properties

    # In a MultiGraph, each edge has a unique `(row, col, key)` key.
    # By default, `key` is 0 if possible, else 1, else 2, etc.
    # This key can be any hashable Python object in NetworkX.
    # We don't use a dict for our data structure here, because
    # that would require a `(row, col, key)` key.
    # Instead, we keep `edge_keys` and/or `edge_indices`.
    # `edge_keys` is the list of Python objects for each edge.
    # `edge_indices` is for the common case of default multiedge keys,
    # in which case we can store it as a cupy array.
    # `edge_indices` is generally preferred. It is possible to provide
    # both where edge_indices is the default and edge_keys is anything.
    # It is also possible for them both to be None, which means the
    # default edge indices has not yet been calculated.
    edge_indices: cp.ndarray[IndexValue] | None
    edge_keys: list[EdgeKey] | None

    ####################
    # Creation methods #
    ####################

    @classmethod
    def from_coo(
        cls,
        N: int,
        row_indices: cp.ndarray[IndexValue],
        col_indices: cp.ndarray[IndexValue],
        edge_indices: cp.ndarray[IndexValue] | None = None,
        edge_values: dict[AttrKey, cp.ndarray[EdgeValue]] | None = None,
        edge_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        node_values: dict[AttrKey, cp.ndarray[NodeValue]] | None = None,
        node_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        *,
        key_to_id: dict[NodeKey, IndexValue] | None = None,
        id_to_key: list[NodeKey] | None = None,
        edge_keys: list[EdgeKey] | None = None,
        **attr,
    ) -> MultiGraph:
        new_graph = super().from_coo(
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
        new_graph.edge_indices = edge_indices
        new_graph.edge_keys = edge_keys
        # Easy and fast sanity checks
        if (
            new_graph.edge_keys is not None
            and len(new_graph.edge_keys) != row_indices.size
        ):
            raise ValueError
        return new_graph

    # TODO:
    # from_csr
    # from_csc
    # from_dcsr
    # from_dcsc

    def __new__(cls, incoming_graph_data=None, multigraph_input=None, **attr) -> Graph:
        # TODO: handle multigraph_input
        if incoming_graph_data is None:
            new_graph = cls.from_coo(0, cp.empty(0, np.int32), cp.empty(0, np.int32))
        elif incoming_graph_data.__class__ is new_graph.__class__:
            new_graph = incoming_graph_data.copy()
        elif incoming_graph_data.__class__ is new_graph.to_networkx_class():
            new_graph = nxcg.from_networkx(incoming_graph_data, preserve_all_attrs=True)
        else:
            raise NotImplementedError
        new_graph.graph.update(attr)
        return new_graph

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
        return True

    @classmethod
    @networkx_api
    def to_directed_class(cls) -> type[nxcg.MultiDiGraph]:
        return nxcg.MultiDiGraph

    @classmethod
    def to_networkx_class(cls) -> type[nx.MultiGraph]:
        return nx.MultiGraph

    @classmethod
    @networkx_api
    def to_undirected_class(cls) -> type[MultiGraph]:
        return MultiGraph

    ##########################
    # NetworkX graph methods #
    ##########################

    @networkx_api
    def clear(self) -> None:
        super().clear()
        self.edge_indices = None
        self.edge_keys = None

    @networkx_api
    def clear_edges(self) -> None:
        super().clear_edges()
        self.edge_indices = None
        self.edge_keys = None

    # TODO:
    # copy
    # get_edge_data

    @networkx_api
    def has_edge(self, u: NodeKey, v: NodeKey, key: EdgeKey | None = None) -> bool:
        if self.key_to_id is not None:
            try:
                u = self.key_to_id[u]
                v = self.key_to_id[v]
            except KeyError:
                return False
        mask = (self.row_indices == u) & (self.col_indices == v)
        if key is None or (self.edge_indices is None and self.edge_keys is None):
            return bool(mask.any())
        if self.edge_keys is None:
            return bool((mask & (self.edge_indices == key)).any())
        indices = cp.nonzero(mask)[0]
        if indices.size == 0:
            return False
        edge_keys = self.edge_keys
        return any(edge_keys[i] == key for i in indices.tolist())

    @networkx_api
    def to_directed(self, as_view: bool = False) -> nxcg.MultiDiGraph:
        return self._copy(as_view, self.to_directed_class())

    @networkx_api
    def to_undirected(self, as_view: bool = False) -> MultiGraph:
        # Does deep copy in networkx
        return self.copy(as_view)

    ###################
    # Private methods #
    ###################

    def _copy(self, as_view: bool, cls: type[Graph], reverse: bool = False):
        # DRY warning: see also Graph._copy
        indptr = self.indptr
        row_indices = self.row_indices
        col_indices = self.col_indices
        edge_indices = self.edge_indices
        edge_values = self.edge_values
        edge_masks = self.edge_masks
        node_values = self.node_values
        node_masks = self.node_masks
        key_to_id = self.key_to_id
        id_to_key = None if key_to_id is None else self._id_to_key
        edge_keys = self.edge_keys
        if not as_view:
            indptr = indptr.copy()
            row_indices = row_indices.copy()
            col_indices = col_indices.copy()
            edge_indices = edge_indices.copy()
            edge_values = {key: val.copy() for key, val in edge_values.items()}
            edge_masks = {key: val.copy() for key, val in edge_masks.items()}
            node_values = {key: val.copy() for key, val in node_values.items()}
            node_masks = {key: val.copy() for key, val in node_masks.items()}
            if key_to_id is not None:
                key_to_id = key_to_id.copy()
                if id_to_key is not None:
                    id_to_key = id_to_key.copy()
            if edge_keys is not None:
                edge_keys = edge_keys.copy()
        if reverse:
            row_indices, col_indices = col_indices, row_indices
        rv = cls.from_coo(
            indptr,
            row_indices,
            col_indices,
            edge_indices,
            edge_values,
            edge_masks,
            node_values,
            node_masks,
            key_to_id=key_to_id,
            id_to_key=id_to_key,
            edge_keys=edge_keys,
        )
        if as_view:
            rv.graph = self.graph
        else:
            rv.graph.update(deepcopy(self.graph))
        return rv
