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

import operator as op
from copy import deepcopy
from typing import TYPE_CHECKING

import cupy as cp
import networkx as nx
import numpy as np
import pylibcugraph as plc

import nx_cugraph as nxcg

from ..utils import index_dtype

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Iterator
    from typing import ClassVar

    from nx_cugraph.typing import (
        AttrKey,
        Dtype,
        EdgeTuple,
        EdgeValue,
        IndexValue,
        NodeKey,
        NodeValue,
        any_ndarray,
    )

__all__ = ["Graph"]

networkx_api = nxcg.utils.decorators.networkx_class(nx.Graph)


class Graph:
    # Tell networkx to dispatch calls with this object to nx-cugraph
    __networkx_backend__: ClassVar[str] = "cugraph"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "cugraph"  # nx <3.2

    # networkx properties
    graph: dict
    graph_attr_dict_factory: ClassVar[type] = dict

    # Not networkx properties
    # We store edge data in COO format with {src,dst}_indices and edge_values.
    src_indices: cp.ndarray[IndexValue]
    dst_indices: cp.ndarray[IndexValue]
    edge_values: dict[AttrKey, cp.ndarray[EdgeValue]]
    edge_masks: dict[AttrKey, cp.ndarray[bool]]
    node_values: dict[AttrKey, any_ndarray[NodeValue]]
    node_masks: dict[AttrKey, any_ndarray[bool]]
    key_to_id: dict[NodeKey, IndexValue] | None
    _id_to_key: list[NodeKey] | None
    _N: int
    _node_ids: cp.ndarray[IndexValue] | None  # holds plc.SGGraph.vertices_array data

    # Used by graph._get_plc_graph
    _plc_type_map: ClassVar[dict[np.dtype, np.dtype]] = {
        # signed int
        np.dtype(np.int8): np.dtype(np.float32),
        np.dtype(np.int16): np.dtype(np.float32),
        np.dtype(np.int32): np.dtype(np.float64),
        np.dtype(np.int64): np.dtype(np.float64),  # raise if abs(x) > 2**53
        # unsigned int
        np.dtype(np.uint8): np.dtype(np.float32),
        np.dtype(np.uint16): np.dtype(np.float32),
        np.dtype(np.uint32): np.dtype(np.float64),
        np.dtype(np.uint64): np.dtype(np.float64),  # raise if x > 2**53
        # other
        np.dtype(np.bool_): np.dtype(np.float32),
        np.dtype(np.float16): np.dtype(np.float32),
    }
    _plc_allowed_edge_types: ClassVar[set[np.dtype]] = {
        np.dtype(np.float32),
        np.dtype(np.float64),
    }

    ####################
    # Creation methods #
    ####################

    @classmethod
    def from_coo(
        cls,
        N: int,
        src_indices: cp.ndarray[IndexValue],
        dst_indices: cp.ndarray[IndexValue],
        edge_values: dict[AttrKey, cp.ndarray[EdgeValue]] | None = None,
        edge_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        node_values: dict[AttrKey, any_ndarray[NodeValue]] | None = None,
        node_masks: dict[AttrKey, any_ndarray[bool]] | None = None,
        *,
        key_to_id: dict[NodeKey, IndexValue] | None = None,
        id_to_key: list[NodeKey] | None = None,
        **attr,
    ) -> Graph:
        new_graph = object.__new__(cls)
        new_graph.src_indices = src_indices
        new_graph.dst_indices = dst_indices
        new_graph.edge_values = {} if edge_values is None else dict(edge_values)
        new_graph.edge_masks = {} if edge_masks is None else dict(edge_masks)
        new_graph.node_values = {} if node_values is None else dict(node_values)
        new_graph.node_masks = {} if node_masks is None else dict(node_masks)
        new_graph.key_to_id = None if key_to_id is None else dict(key_to_id)
        new_graph._id_to_key = None if id_to_key is None else list(id_to_key)
        new_graph._N = op.index(N)  # Ensure N is integral
        new_graph._node_ids = None
        new_graph.graph = new_graph.graph_attr_dict_factory()
        new_graph.graph.update(attr)
        size = new_graph.src_indices.size
        # Easy and fast sanity checks
        if size != new_graph.dst_indices.size:
            raise ValueError
        for edge_attr in ["edge_values", "edge_masks"]:
            if datadict := getattr(new_graph, edge_attr):
                for key, val in datadict.items():
                    if val.shape[0] != size:
                        raise ValueError(key)
        for node_attr in ["node_values", "node_masks"]:
            if datadict := getattr(new_graph, node_attr):
                for key, val in datadict.items():
                    if val.shape[0] != N:
                        raise ValueError(key)
        if new_graph.key_to_id is not None and len(new_graph.key_to_id) != N:
            raise ValueError
        if new_graph._id_to_key is not None and len(new_graph._id_to_key) != N:
            raise ValueError
        if new_graph._id_to_key is not None and new_graph.key_to_id is None:
            try:
                new_graph.key_to_id = dict(zip(new_graph._id_to_key, range(N)))
            except TypeError as exc:
                raise ValueError("Bad type of a node value") from exc
        if new_graph.src_indices.dtype != index_dtype:
            src_indices = new_graph.src_indices.astype(index_dtype)
            if not (new_graph.src_indices == src_indices).all():
                raise ValueError(
                    f"Unable to convert src_indices to {src_indices.dtype.name} "
                    f"(got {new_graph.src_indices.dtype.name})."
                )
            new_graph.src_indices = src_indices
        if new_graph.dst_indices.dtype != index_dtype:
            dst_indices = new_graph.dst_indices.astype(index_dtype)
            if not (new_graph.dst_indices == dst_indices).all():
                raise ValueError(
                    f"Unable to convert dst_indices to {dst_indices.dtype.name} "
                    f"(got {new_graph.dst_indices.dtype.name})."
                )
            new_graph.dst_indices = dst_indices

        # If the graph contains isolates, plc.SGGraph() must be passed a value
        # for vertices_array that contains every vertex ID, since the
        # src/dst_indices arrays will not contain IDs for isolates. Create this
        # only if needed. Like src/dst_indices, the _node_ids array must be
        # maintained for the lifetime of the plc.SGGraph
        isolates = nxcg.algorithms.isolate._isolates(new_graph)
        if len(isolates) > 0:
            new_graph._node_ids = cp.arange(new_graph._N, dtype=index_dtype)

        return new_graph

    @classmethod
    def from_csr(
        cls,
        indptr: cp.ndarray[IndexValue],
        dst_indices: cp.ndarray[IndexValue],
        edge_values: dict[AttrKey, cp.ndarray[EdgeValue]] | None = None,
        edge_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        node_values: dict[AttrKey, any_ndarray[NodeValue]] | None = None,
        node_masks: dict[AttrKey, any_ndarray[bool]] | None = None,
        *,
        key_to_id: dict[NodeKey, IndexValue] | None = None,
        id_to_key: list[NodeKey] | None = None,
        **attr,
    ) -> Graph:
        N = indptr.size - 1
        src_indices = cp.array(
            # cp.repeat is slow to use here, so use numpy instead
            np.repeat(np.arange(N, dtype=index_dtype), cp.diff(indptr).get())
        )
        return cls.from_coo(
            N,
            src_indices,
            dst_indices,
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
        indptr: cp.ndarray[IndexValue],
        src_indices: cp.ndarray[IndexValue],
        edge_values: dict[AttrKey, cp.ndarray[EdgeValue]] | None = None,
        edge_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        node_values: dict[AttrKey, any_ndarray[NodeValue]] | None = None,
        node_masks: dict[AttrKey, any_ndarray[bool]] | None = None,
        *,
        key_to_id: dict[NodeKey, IndexValue] | None = None,
        id_to_key: list[NodeKey] | None = None,
        **attr,
    ) -> Graph:
        N = indptr.size - 1
        dst_indices = cp.array(
            # cp.repeat is slow to use here, so use numpy instead
            np.repeat(np.arange(N, dtype=index_dtype), cp.diff(indptr).get())
        )
        return cls.from_coo(
            N,
            src_indices,
            dst_indices,
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
        compressed_srcs: cp.ndarray[IndexValue],
        indptr: cp.ndarray[IndexValue],
        dst_indices: cp.ndarray[IndexValue],
        edge_values: dict[AttrKey, cp.ndarray[EdgeValue]] | None = None,
        edge_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        node_values: dict[AttrKey, any_ndarray[NodeValue]] | None = None,
        node_masks: dict[AttrKey, any_ndarray[bool]] | None = None,
        *,
        key_to_id: dict[NodeKey, IndexValue] | None = None,
        id_to_key: list[NodeKey] | None = None,
        **attr,
    ) -> Graph:
        src_indices = cp.array(
            # cp.repeat is slow to use here, so use numpy instead
            np.repeat(compressed_srcs.get(), cp.diff(indptr).get())
        )
        return cls.from_coo(
            N,
            src_indices,
            dst_indices,
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
        compressed_dsts: cp.ndarray[IndexValue],
        indptr: cp.ndarray[IndexValue],
        src_indices: cp.ndarray[IndexValue],
        edge_values: dict[AttrKey, cp.ndarray[EdgeValue]] | None = None,
        edge_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        node_values: dict[AttrKey, any_ndarray[NodeValue]] | None = None,
        node_masks: dict[AttrKey, any_ndarray[bool]] | None = None,
        *,
        key_to_id: dict[NodeKey, IndexValue] | None = None,
        id_to_key: list[NodeKey] | None = None,
        **attr,
    ) -> Graph:
        dst_indices = cp.array(
            # cp.repeat is slow to use here, so use numpy instead
            np.repeat(compressed_dsts.get(), cp.diff(indptr).get())
        )
        return cls.from_coo(
            N,
            src_indices,
            dst_indices,
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
            new_graph = cls.from_coo(
                0, cp.empty(0, index_dtype), cp.empty(0, index_dtype)
            )
        elif incoming_graph_data.__class__ is cls:
            new_graph = incoming_graph_data.copy()
        elif incoming_graph_data.__class__ is cls.to_networkx_class():
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
        return False

    @classmethod
    @networkx_api
    def to_directed_class(cls) -> type[nxcg.DiGraph]:
        return nxcg.DiGraph

    @classmethod
    def to_networkx_class(cls) -> type[nx.Graph]:
        return nx.Graph

    @classmethod
    @networkx_api
    def to_undirected_class(cls) -> type[Graph]:
        return Graph

    ##############
    # Properties #
    ##############

    @property
    def edge_dtypes(self) -> dict[AttrKey, Dtype]:
        return {key: val.dtype for key, val in self.edge_values.items()}

    @property
    def node_dtypes(self) -> dict[AttrKey, Dtype]:
        return {key: val.dtype for key, val in self.node_values.items()}

    @property
    def id_to_key(self) -> [NodeKey] | None:
        if self.key_to_id is None:
            return None
        if self._id_to_key is None:
            self._id_to_key = sorted(self.key_to_id, key=self.key_to_id.__getitem__)
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
    def add_nodes_from(self, nodes_for_adding: Iterable[NodeKey], **attr) -> None:
        if self._N != 0:
            raise NotImplementedError(
                "add_nodes_from is not implemented for graph that already has nodes."
            )
        G = self.to_networkx_class()()
        G.add_nodes_from(nodes_for_adding, **attr)
        G = nxcg.from_networkx(G, preserve_node_attrs=True)
        self._become(G)

    @networkx_api
    def clear(self) -> None:
        self.edge_values.clear()
        self.edge_masks.clear()
        self.node_values.clear()
        self.node_masks.clear()
        self.graph.clear()
        self.src_indices = cp.empty(0, self.src_indices.dtype)
        self.dst_indices = cp.empty(0, self.dst_indices.dtype)
        self._N = 0
        self._node_ids = None
        self.key_to_id = None
        self._id_to_key = None

    @networkx_api
    def clear_edges(self) -> None:
        self.edge_values.clear()
        self.edge_masks.clear()
        self.src_indices = cp.empty(0, self.src_indices.dtype)
        self.dst_indices = cp.empty(0, self.dst_indices.dtype)

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
        else:
            try:
                if u < 0 or v < 0 or u >= self._N or v >= self._N:
                    return default
            except TypeError:
                return default
        index = cp.nonzero((self.src_indices == u) & (self.dst_indices == v))[0]
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
        return bool(((self.src_indices == u) & (self.dst_indices == v)).any())

    def _neighbors(self, n: NodeKey) -> cp.ndarray[NodeValue]:
        if n not in self:
            hash(n)  # To raise TypeError if appropriate
            raise nx.NetworkXError(
                f"The node {n} is not in the {self.__class__.__name__.lower()}."
            )
        if self.key_to_id is not None:
            n = self.key_to_id[n]
        nbrs = self.dst_indices[self.src_indices == n]
        if self.is_multigraph():
            nbrs = cp.unique(nbrs)
        return nbrs

    @networkx_api
    def neighbors(self, n: NodeKey) -> Iterator[NodeKey]:
        nbrs = self._neighbors(n)
        return iter(self._nodeiter_to_iter(nbrs.tolist()))

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
        # If no self-edges, then `self.src_indices.size // 2`
        return int(cp.count_nonzero(self.src_indices <= self.dst_indices))

    @networkx_api
    def to_directed(self, as_view: bool = False) -> nxcg.DiGraph:
        return self._copy(as_view, self.to_directed_class())

    @networkx_api
    def to_undirected(self, as_view: bool = False) -> Graph:
        # Does deep copy in networkx
        return self._copy(as_view, self.to_undirected_class())

    # Not implemented...
    # adj, adjacency, add_edge, add_edges_from, add_node,
    # add_nodes_from, add_weighted_edges_from, degree,
    # edge_subgraph, edges, neighbors, nodes, remove_edge,
    # remove_edges_from, remove_node, remove_nodes_from, subgraph, update

    ###################
    # Private methods #
    ###################

    def _copy(self, as_view: bool, cls: type[Graph], reverse: bool = False):
        # DRY warning: see also MultiGraph._copy
        src_indices = self.src_indices
        dst_indices = self.dst_indices
        edge_values = self.edge_values
        edge_masks = self.edge_masks
        node_values = self.node_values
        node_masks = self.node_masks
        key_to_id = self.key_to_id
        id_to_key = None if key_to_id is None else self._id_to_key
        if not as_view:
            src_indices = src_indices.copy()
            dst_indices = dst_indices.copy()
            edge_values = {key: val.copy() for key, val in edge_values.items()}
            edge_masks = {key: val.copy() for key, val in edge_masks.items()}
            node_values = {key: val.copy() for key, val in node_values.items()}
            node_masks = {key: val.copy() for key, val in node_masks.items()}
            if key_to_id is not None:
                key_to_id = key_to_id.copy()
                if id_to_key is not None:
                    id_to_key = id_to_key.copy()
        if reverse:
            src_indices, dst_indices = dst_indices, src_indices
        rv = cls.from_coo(
            self._N,
            src_indices,
            dst_indices,
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
        self,
        edge_attr: AttrKey | None = None,
        edge_default: EdgeValue | None = None,
        edge_dtype: Dtype | None = None,
        *,
        store_transposed: bool = False,
        switch_indices: bool = False,
        edge_array: cp.ndarray[EdgeValue] | None = None,
        symmetrize: str | None = None,
    ):
        if edge_array is not None or edge_attr is None:
            pass
        elif edge_attr not in self.edge_values:
            if edge_default is None:
                raise KeyError("Graph has no edge attribute {edge_attr!r}")
            # If we were given a default edge value, then it's probably okay to
            # use None for the edge_array if we don't have this edge attribute.
        elif edge_attr not in self.edge_masks:
            edge_array = self.edge_values[edge_attr]
        elif not self.edge_masks[edge_attr].all():
            if edge_default is None:
                raise NotImplementedError(
                    "Missing edge attributes is not yet implemented"
                )
            edge_array = cp.where(
                self.edge_masks[edge_attr], self.edge_values[edge_attr], edge_default
            )
        else:
            # Mask is all True; don't need anymore
            del self.edge_masks[edge_attr]
            edge_array = self.edge_values[edge_attr]
        if edge_array is not None:
            if edge_dtype is not None:
                edge_dtype = np.dtype(edge_dtype)
                if edge_array.dtype != edge_dtype:
                    edge_array = edge_array.astype(edge_dtype)
            # PLC doesn't handle int edge weights right now, so cast int to float
            if edge_array.dtype in self._plc_type_map:
                if edge_array.dtype == np.int64:
                    if (val := edge_array.max().tolist()) > 2**53:
                        raise ValueError(
                            f"Integer value of value is too large (> 2**53): {val}; "
                            "pylibcugraph only supports float16 and float32 dtypes."
                        )
                    if (val := edge_array.min().tolist()) < -(2**53):
                        raise ValueError(
                            f"Integer value of value is small large (< -2**53): {val}; "
                            "pylibcugraph only supports float16 and float32 dtypes."
                        )
                elif (
                    edge_array.dtype == np.uint64 and edge_array.max().tolist() > 2**53
                ):
                    raise ValueError(
                        f"Integer value of value is too large (> 2**53): {val}; "
                        "pylibcugraph only supports float16 and float32 dtypes."
                    )
                # Consider warning here if we add algorithms that may
                # introduce roundoff errors when using floats as ints.
                edge_array = edge_array.astype(self._plc_type_map[edge_array.dtype])
            elif edge_array.dtype not in self._plc_allowed_edge_types:
                raise TypeError(edge_array.dtype)
        # Should we cache PLC graph?
        src_indices = self.src_indices
        dst_indices = self.dst_indices
        if switch_indices:
            src_indices, dst_indices = dst_indices, src_indices
        if symmetrize is not None:
            if edge_array is not None:
                raise NotImplementedError(
                    "edge_array must be None when symmetrizing the graph"
                )
            N = self._N
            # Upcast to int64 so indices don't overflow
            src_dst = N * src_indices.astype(np.int64) + dst_indices
            src_dst_T = src_indices + N * dst_indices.astype(np.int64)
            if symmetrize == "union":
                src_dst_new = cp.union1d(src_dst, src_dst_T)
            elif symmetrize == "intersection":
                src_dst_new = cp.intersect1d(src_dst, src_dst_T)
            else:
                raise ValueError(
                    f'symmetrize must be "union" or "intersection"; got "{symmetrize}"'
                )
            src_indices, dst_indices = cp.divmod(src_dst_new, N)
            src_indices = src_indices.astype(index_dtype)
            dst_indices = dst_indices.astype(index_dtype)

        return plc.SGGraph(
            resource_handle=plc.ResourceHandle(),
            graph_properties=plc.GraphProperties(
                is_multigraph=self.is_multigraph() and symmetrize is None,
                is_symmetric=not self.is_directed() or symmetrize is not None,
            ),
            src_or_offset_array=src_indices,
            dst_or_index_array=dst_indices,
            weight_array=edge_array,
            store_transposed=store_transposed,
            renumber=False,
            do_expensive_check=False,
            vertices_array=self._node_ids,
        )

    def _sort_edge_indices(self, primary="src"):
        # DRY warning: see also MultiGraph._sort_edge_indices
        if primary == "src":
            stacked = cp.vstack((self.dst_indices, self.src_indices))
        elif primary == "dst":
            stacked = cp.vstack((self.src_indices, self.dst_indices))
        else:
            raise ValueError(
                f'Bad `primary` argument; expected "src" or "dst", got {primary!r}'
            )
        indices = cp.lexsort(stacked)
        if (cp.diff(indices) > 0).all():
            # Already sorted
            return
        self.src_indices = self.src_indices[indices]
        self.dst_indices = self.dst_indices[indices]
        self.edge_values.update(
            {key: val[indices] for key, val in self.edge_values.items()}
        )
        self.edge_masks.update(
            {key: val[indices] for key, val in self.edge_masks.items()}
        )

    def _become(self, other: Graph):
        if self.__class__ is not other.__class__:
            raise TypeError(
                "Attempting to update graph inplace with graph of different type!"
            )
        self.clear()
        edge_values = self.edge_values
        edge_masks = self.edge_masks
        node_values = self.node_values
        node_masks = self.node_masks
        graph = self.graph
        edge_values.update(other.edge_values)
        edge_masks.update(other.edge_masks)
        node_values.update(other.node_values)
        node_masks.update(other.node_masks)
        graph.update(other.graph)
        self.__dict__.update(other.__dict__)
        self.edge_values = edge_values
        self.edge_masks = edge_masks
        self.node_values = node_values
        self.node_masks = node_masks
        self.graph = graph
        return self

    def _degrees_array(self, *, ignore_selfloops=False):
        src_indices = self.src_indices
        dst_indices = self.dst_indices
        if ignore_selfloops:
            not_selfloops = src_indices != dst_indices
            src_indices = src_indices[not_selfloops]
            if self.is_directed():
                dst_indices = dst_indices[not_selfloops]
        if src_indices.size == 0:
            return cp.zeros(self._N, dtype=np.int64)
        degrees = cp.bincount(src_indices, minlength=self._N)
        if self.is_directed():
            degrees += cp.bincount(dst_indices, minlength=self._N)
        return degrees

    _in_degrees_array = _degrees_array
    _out_degrees_array = _degrees_array

    # Data conversions
    def _nodekeys_to_nodearray(self, nodes: Iterable[NodeKey]) -> cp.array[IndexValue]:
        if self.key_to_id is None:
            return cp.fromiter(nodes, dtype=index_dtype)
        return cp.fromiter(map(self.key_to_id.__getitem__, nodes), dtype=index_dtype)

    def _nodeiter_to_iter(self, node_ids: Iterable[IndexValue]) -> Iterable[NodeKey]:
        """Convert an iterable of node IDs to an iterable of node keys."""
        if (id_to_key := self.id_to_key) is not None:
            return map(id_to_key.__getitem__, node_ids)
        return node_ids

    def _nodearray_to_list(self, node_ids: cp.ndarray[IndexValue]) -> list[NodeKey]:
        if self.key_to_id is None:
            return node_ids.tolist()
        return list(self._nodeiter_to_iter(node_ids.tolist()))

    def _list_to_nodearray(self, nodes: list[NodeKey]) -> cp.ndarray[IndexValue]:
        if (key_to_id := self.key_to_id) is not None:
            nodes = [key_to_id[node] for node in nodes]
        return cp.array(nodes, dtype=index_dtype)

    def _nodearray_to_set(self, node_ids: cp.ndarray[IndexValue]) -> set[NodeKey]:
        if self.key_to_id is None:
            return set(node_ids.tolist())
        return set(self._nodeiter_to_iter(node_ids.tolist()))

    def _nodearray_to_dict(
        self, values: cp.ndarray[NodeValue]
    ) -> dict[NodeKey, NodeValue]:
        it = enumerate(values.tolist())
        if (id_to_key := self.id_to_key) is not None:
            return {id_to_key[key]: val for key, val in it}
        return dict(it)

    def _nodearrays_to_dict(
        self, node_ids: cp.ndarray[IndexValue], values: any_ndarray[NodeValue]
    ) -> dict[NodeKey, NodeValue]:
        it = zip(node_ids.tolist(), values.tolist())
        if (id_to_key := self.id_to_key) is not None:
            return {id_to_key[key]: val for key, val in it}
        return dict(it)

    def _edgearrays_to_dict(
        self,
        src_ids: cp.ndarray[IndexValue],
        dst_ids: cp.ndarray[IndexValue],
        values: cp.ndarray[EdgeValue],
    ) -> dict[EdgeTuple, EdgeValue]:
        it = zip(zip(src_ids.tolist(), dst_ids.tolist()), values.tolist())
        if (id_to_key := self.id_to_key) is not None:
            return {
                (id_to_key[src_id], id_to_key[dst_id]): val
                for (src_id, dst_id), val in it
            }
        return dict(it)

    def _dict_to_nodearrays(
        self,
        d: dict[NodeKey, NodeValue],
        dtype: Dtype | None = None,
    ) -> tuple[cp.ndarray[IndexValue], cp.ndarray[NodeValue]]:
        if self.key_to_id is None:
            indices_iter = d
        else:
            indices_iter = map(self.key_to_id.__getitem__, d)
        node_ids = cp.fromiter(indices_iter, index_dtype)
        if dtype is None:
            values = cp.array(list(d.values()))
        else:
            values = cp.fromiter(d.values(), dtype)
        return node_ids, values

    def _dict_to_nodearray(
        self,
        d: dict[NodeKey, NodeValue] | cp.ndarray[NodeValue],
        default: NodeValue | None = None,
        dtype: Dtype | None = None,
    ) -> cp.ndarray[NodeValue]:
        if isinstance(d, cp.ndarray):
            if d.shape[0] != len(self):
                raise ValueError
            if dtype is not None and d.dtype != dtype:
                return d.astype(dtype)
            return d
        if default is None:
            val_iter = map(d.__getitem__, self)
        else:
            val_iter = (d.get(node, default) for node in self)
        if dtype is None:
            return cp.array(list(val_iter))
        return cp.fromiter(val_iter, dtype)
