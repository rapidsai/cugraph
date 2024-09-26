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
from networkx.classes.graph import (
    _CachedPropertyResetterAdj,
    _CachedPropertyResetterNode,
)

import nx_cugraph as nxcg
from nx_cugraph import _nxver

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

__all__ = ["CudaGraph", "Graph"]

networkx_api = nxcg.utils.decorators.networkx_class(nx.Graph)

# The "everything" cache key is an internal implementation detail of NetworkX
# that may change between releases.
if _nxver < (3, 4):
    _CACHE_KEY = (
        True,  # Include all edge values
        True,  # Include all node values
        True,  # Include `.graph` attributes
    )
else:
    _CACHE_KEY = (
        True,  # Include all edge values
        True,  # Include all node values
        # `.graph` attributes are always included now
    )

# Use to indicate when a full conversion to GPU failed so we don't try again.
_CANT_CONVERT_TO_GPU = "_CANT_CONVERT_TO_GPU"


# `collections.UserDict` was the preferred way to subclass dict, but now
# subclassing dict directly is much better supported and should work here.
# This class should only be necessary if the user clears the cache manually.
class _GraphCache(dict):
    """Cache that ensures Graph will reify into a NetworkX graph when cleared."""

    _graph: Graph

    def __init__(self, graph: Graph):
        self._graph = graph

    def clear(self) -> None:
        self._graph._reify_networkx()
        super().clear()


class Graph(nx.Graph):
    # Tell networkx to dispatch calls with this object to nx-cugraph
    __networkx_backend__: ClassVar[str] = "cugraph"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "cugraph"  # nx <3.2

    # Core attributes of NetowkrX graphs that will be copied and cleared as appropriate.
    # These attributes comprise the edge and node data model for NetworkX graphs.
    _nx_attrs = ("_node", "_adj")

    # Allow networkx dispatch machinery to cache conversions.
    # This means we should clear the cache if we ever mutate the object!
    __networkx_cache__: _GraphCache | None

    # networkx properties
    graph: dict
    # Should we declare type annotations for the rest?

    # Properties that trigger copying to the CPU
    def _prepare_setter(self):
        """Be careful when setting private attributes which may be used during init."""
        if (
            # If not present, then this must be in init
            any(attr not in self.__dict__ for attr in self._nx_attrs)
            # Already on the CPU
            or not any(self.__dict__[attr] is None for attr in self._nx_attrs)
        ):
            return
        if self._is_on_gpu:
            # Copy from GPU to CPU
            self._reify_networkx()
            return
        # Default values
        for attr in self._nx_attrs:
            if self.__dict__[attr] is None:
                if attr == "_succ":
                    self.__dict__[attr] = self.__dict__["_adj"]
                else:
                    self.__dict__[attr] = {}

    @property
    @networkx_api
    def _node(self):
        if (node := self.__dict__["_node"]) is None:
            self._reify_networkx()
            node = self.__dict__["_node"]
        return node

    @_node.setter
    def _node(self, val):
        self._prepare_setter()
        _CachedPropertyResetterNode.__set__(None, self, val)
        if cache := getattr(self, "__networkx_cache__", None):
            cache.clear()

    @property
    @networkx_api
    def _adj(self):
        if (adj := self.__dict__["_adj"]) is None:
            self._reify_networkx()
            adj = self.__dict__["_adj"]
        return adj

    @_adj.setter
    def _adj(self, val):
        self._prepare_setter()
        _CachedPropertyResetterAdj.__set__(None, self, val)
        if cache := getattr(self, "__networkx_cache__", None):
            cache.clear()

    @property
    def _is_on_gpu(self) -> bool:
        """Whether the full graph is on device (in the cache).

        This returns False when only a subset of the graph (such as only
        edge indices and edge attribute) is on device.

        The graph may be on host (CPU) and device (GPU) at the same time.
        """
        cache = getattr(self, "__networkx_cache__", None)
        if not cache:
            return False
        return _CACHE_KEY in cache.get("backends", {}).get("cugraph", {})

    @property
    def _is_on_cpu(self) -> bool:
        """Whether the graph is on host as a NetworkX graph.

        This means the core data structures that comprise a NetworkX graph
        (such as ``G._node`` and ``G._adj``) are present.

        The graph may be on host (CPU) and device (GPU) at the same time.
        """
        return self.__dict__["_node"] is not None

    @property
    def _cudagraph(self):
        """Return the full ``CudaGraph`` on device, computing if necessary, or None."""
        nx_cache = getattr(self, "__networkx_cache__", None)
        if nx_cache is None:
            nx_cache = {}
        elif _CANT_CONVERT_TO_GPU in nx_cache:
            return None
        cache = nx_cache.setdefault("backends", {}).setdefault("cugraph", {})
        if (Gcg := cache.get(_CACHE_KEY)) is not None:
            if isinstance(Gcg, Graph):
                # This shouldn't happen during normal use, but be extra-careful anyway
                return Gcg._cudagraph
            return Gcg
        if self.__dict__["_node"] is None:
            raise RuntimeError(
                f"{type(self).__name__} cannot be converted to the GPU, because it is "
                "not on the CPU! This is not supposed to be possible. If you believe "
                "you have found a bug, please report a minimum reproducible example to "
                "https://github.com/rapidsai/cugraph/issues/new/choose"
            )
        try:
            Gcg = nxcg.from_networkx(
                self, preserve_edge_attrs=True, preserve_node_attrs=True
            )
        except Exception:
            # Should we warn that the full graph can't be on GPU?
            nx_cache[_CANT_CONVERT_TO_GPU] = True
            return None
        Gcg.graph = self.graph
        cache[_CACHE_KEY] = Gcg
        return Gcg

    @_cudagraph.setter
    def _cudagraph(self, val, *, clear_cpu=True):
        """Set the full ``CudaGraph`` for this graph, or remove from device if None."""
        if (cache := getattr(self, "__networkx_cache__", None)) is None:
            # Should we warn?
            return
        # TODO: pay close attention to when we should clear the cache, since
        # this may or may not be a mutation.
        cache = cache.setdefault("backends", {}).setdefault("cugraph", {})
        if val is None:
            cache.pop(_CACHE_KEY, None)
        else:
            self.graph = val.graph
            cache[_CACHE_KEY] = val
            if clear_cpu:
                for key in self._nx_attrs:
                    self.__dict__[key] = None

    @nx.Graph.name.setter
    def name(self, s):
        # Don't clear the cache when setting the name, since `.graph` is shared.
        # There is a very small risk here for the cache to become (slightly)
        # insconsistent if graphs from other backends are cached.
        self.graph["name"] = s

    @classmethod
    @networkx_api
    def is_directed(cls) -> bool:
        return False

    @classmethod
    @networkx_api
    def is_multigraph(cls) -> bool:
        return False

    @classmethod
    def to_cudagraph_class(cls) -> type[CudaGraph]:
        return CudaGraph

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

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.__networkx_cache__ = _GraphCache(self)

    def _reify_networkx(self) -> None:
        """Copy graph to host (CPU) if necessary."""
        if self.__dict__["_node"] is None:
            # After we make this into an nx graph, we rely on the cache being correct
            Gcg = self._cudagraph
            G = nxcg.to_networkx(Gcg)
            for key in self._nx_attrs:
                self.__dict__[key] = G.__dict__[key]

    def _become(self, other: Graph):
        if self.__class__ is not other.__class__:
            raise TypeError(
                "Attempting to update graph inplace with graph of different type!"
            )
        # Begin with the simplest implementation; do we need to do more?
        self.__dict__.update(other.__dict__)
        return self

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
        use_compat_graph: bool | None = None,
        **attr,
    ) -> Graph | CudaGraph:
        new_graph = object.__new__(cls.to_cudagraph_class())
        new_graph.__networkx_cache__ = {}
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
        if use_compat_graph or use_compat_graph is None and issubclass(cls, Graph):
            new_graph = new_graph._to_compat_graph()
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
        use_compat_graph: bool | None = None,
        **attr,
    ) -> Graph | CudaGraph:
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
            use_compat_graph=use_compat_graph,
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
        use_compat_graph: bool | None = None,
        **attr,
    ) -> Graph | CudaGraph:
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
            use_compat_graph=use_compat_graph,
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
        use_compat_graph: bool | None = None,
        **attr,
    ) -> Graph | CudaGraph:
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
            use_compat_graph=use_compat_graph,
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
        use_compat_graph: bool | None = None,
        **attr,
    ) -> Graph | CudaGraph:
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
            use_compat_graph=use_compat_graph,
            **attr,
        )


class CudaGraph:
    # Tell networkx to dispatch calls with this object to nx-cugraph
    __networkx_backend__: ClassVar[str] = "cugraph"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "cugraph"  # nx <3.2

    # Allow networkx dispatch machinery to cache conversions.
    # This means we should clear the cache if we ever mutate the object!
    __networkx_cache__: dict | None

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

    from_coo = classmethod(Graph.from_coo.__func__)
    from_csr = classmethod(Graph.from_csr.__func__)
    from_csc = classmethod(Graph.from_csc.__func__)
    from_dcsr = classmethod(Graph.from_dcsr.__func__)
    from_dcsc = classmethod(Graph.from_dcsc.__func__)

    def __new__(cls, incoming_graph_data=None, **attr) -> CudaGraph:
        if incoming_graph_data is None:
            new_graph = cls.from_coo(
                0,
                cp.empty(0, index_dtype),
                cp.empty(0, index_dtype),
                use_compat_graph=False,
            )
        elif incoming_graph_data.__class__ is cls:
            new_graph = incoming_graph_data.copy()
        elif incoming_graph_data.__class__ is cls.to_networkx_class():
            new_graph = nxcg.from_networkx(incoming_graph_data, preserve_all_attrs=True)
        else:
            raise NotImplementedError
        new_graph.graph.update(attr)
        # We could return Graph here (if configured), but let's not for now
        return new_graph

    #################
    # Class methods #
    #################

    is_directed = classmethod(Graph.is_directed.__func__)
    is_multigraph = classmethod(Graph.is_multigraph.__func__)
    to_cudagraph_class = classmethod(Graph.to_cudagraph_class.__func__)
    to_networkx_class = classmethod(Graph.to_networkx_class.__func__)

    @classmethod
    @networkx_api
    def to_directed_class(cls) -> type[nxcg.CudaDiGraph]:
        return nxcg.CudaDiGraph

    @classmethod
    @networkx_api
    def to_undirected_class(cls) -> type[CudaGraph]:
        return CudaGraph

    @classmethod
    def _to_compat_graph_class(cls) -> type[Graph]:
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
        if cache := self.__networkx_cache__:
            cache.clear()

    @networkx_api
    def clear_edges(self) -> None:
        self.edge_values.clear()
        self.edge_masks.clear()
        self.src_indices = cp.empty(0, self.src_indices.dtype)
        self.dst_indices = cp.empty(0, self.dst_indices.dtype)
        if cache := self.__networkx_cache__:
            cache.clear()

    @networkx_api
    def copy(self, as_view: bool = False) -> CudaGraph:
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
    def to_directed(self, as_view: bool = False) -> nxcg.CudaDiGraph:
        return self._copy(as_view, self.to_directed_class())

    @networkx_api
    def to_undirected(self, as_view: bool = False) -> CudaGraph:
        # Does deep copy in networkx
        return self._copy(as_view, self.to_undirected_class())

    def _to_compat_graph(self) -> Graph:
        rv = self._to_compat_graph_class()()
        rv._cudagraph = self
        return rv

    # Not implemented...
    # adj, adjacency, add_edge, add_edges_from, add_node,
    # add_nodes_from, add_weighted_edges_from, degree,
    # edge_subgraph, edges, neighbors, nodes, remove_edge,
    # remove_edges_from, remove_node, remove_nodes_from, subgraph, update

    ###################
    # Private methods #
    ###################

    def _copy(self, as_view: bool, cls: type[CudaGraph], reverse: bool = False):
        # DRY warning: see also CudaMultiGraph._copy
        src_indices = self.src_indices
        dst_indices = self.dst_indices
        edge_values = self.edge_values
        edge_masks = self.edge_masks
        node_values = self.node_values
        node_masks = self.node_masks
        key_to_id = self.key_to_id
        id_to_key = None if key_to_id is None else self._id_to_key
        if self.__networkx_cache__ is None:
            __networkx_cache__ = None
        elif not reverse and cls is self.__class__:
            __networkx_cache__ = self.__networkx_cache__
        else:
            __networkx_cache__ = {}
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
            if __networkx_cache__ is not None:
                __networkx_cache__ = __networkx_cache__.copy()
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
            use_compat_graph=False,
        )
        if as_view:
            rv.graph = self.graph
        else:
            rv.graph.update(deepcopy(self.graph))
        rv.__networkx_cache__ = __networkx_cache__
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

        # This sets drop_multi_edges=True for non-multigraph input, which means
        # the data in self.src_indices and self.dst_indices may not be
        # identical to that contained in the returned pcl.SGGraph (the returned
        # SGGraph may have fewer edges since duplicates are dropped). Ideally
        # self.src_indices and self.dst_indices would be updated to have
        # duplicate edges removed for non-multigraph instances, but that
        # requires additional code which would be redundant and likely not as
        # performant as the code in PLC.
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
            drop_multi_edges=not self.is_multigraph(),
        )

    def _sort_edge_indices(self, primary="src"):
        # DRY warning: see also CudaMultiGraph._sort_edge_indices
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

    def _become(self, other: CudaGraph):
        if self.__class__ is not other.__class__:
            raise TypeError(
                "Attempting to update graph inplace with graph of different type!"
            )
        self.clear()
        edge_values = self.edge_values
        edge_masks = self.edge_masks
        node_values = self.node_values
        node_masks = self.node_masks
        __networkx_cache__ = self.__networkx_cache__
        graph = self.graph
        edge_values.update(other.edge_values)
        edge_masks.update(other.edge_masks)
        node_values.update(other.node_values)
        node_masks.update(other.node_masks)
        graph.update(other.graph)
        if other.__networkx_cache__ is None:
            __networkx_cache__ = None
        else:
            if __networkx_cache__ is None:
                __networkx_cache__ = {}
            __networkx_cache__.update(other.__networkx_cache__)
        self.__dict__.update(other.__dict__)
        self.edge_values = edge_values
        self.edge_masks = edge_masks
        self.node_values = node_values
        self.node_masks = node_masks
        self.graph = graph
        self.__networkx_cache__ = __networkx_cache__
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
