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

from collections.abc import Hashable, Iterator
from copy import deepcopy
from typing import TYPE_CHECKING, TypeVar

import cugraph_nx as cnx

if TYPE_CHECKING:
    import cupy as cp

NodeType = TypeVar("NodeType", bound=Hashable)
AttrType = TypeVar("AttrType", bound=Hashable)


__all__ = ["Graph"]


class Graph:
    indptr: cp.ndarray
    row_indices: cp.ndarray
    col_indices: cp.ndarray
    edge_values: dict[AttrType, cp.ndarray]
    edge_masks: dict[AttrType, cp.ndarray]
    node_values: dict[AttrType, cp.ndarray]
    node_masks: dict[AttrType, cp.ndarray]
    key_to_id: dict[NodeType, int] | None
    _id_to_key: dict[int, NodeType] | None
    _N: int
    graph: dict

    __networkx_plugin__ = "cugraph"

    graph_attr_dict_factory = dict

    def __init__(
        self,
        indptr,
        row_indices,
        col_indices,
        edge_values,
        edge_masks,
        node_values,
        node_masks,
        *,
        key_to_id,
        id_to_key=None,
        **attr,
    ):
        # TODO: when should we make and wrap a plc.Graph? Can it own these arrays?
        self.indptr = indptr
        self.row_indices = row_indices
        self.col_indices = col_indices
        self.edge_values = edge_values
        self.edge_masks = edge_masks
        self.node_values = node_values
        self.node_masks = node_masks
        self.key_to_id = key_to_id
        self._id_to_key = id_to_key
        self._N = indptr.size - 1
        self.graph = self.graph_attr_dict_factory()
        self.graph.update(attr)

    @property
    def id_to_key(self) -> dict[int, NodeType]:
        if self.key_to_id is None:
            return None
        if self._id_to_key is None:
            self._id_to_key = {val: key for key, val in self._key_to_id.items()}
        return self._id_to_key

    def __iter__(self) -> Iterator:
        if self.key_to_id is not None:
            return iter(self.key_to_id)
        return iter(range(self._N))

    def __contains__(self, n) -> bool:
        if self.key_to_id is not None:
            container = self.key_to_id
        else:
            container = range(self._N)
        try:
            return n in container
        except TypeError:
            return False

    def __len__(self) -> int:
        return self._N

    def number_of_nodes(self) -> int:
        return self._N

    def order(self) -> int:
        return self._N

    def is_multigraph(self) -> bool:
        return False

    def is_directed(self) -> bool:
        return False

    def to_directed_class(self):
        return cnx.DiGraph

    def to_undirected_class(self):
        return Graph

    def to_directed(self, as_view=False) -> cnx.DiGraph:
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
        rv = cnx.DiGraph(
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
