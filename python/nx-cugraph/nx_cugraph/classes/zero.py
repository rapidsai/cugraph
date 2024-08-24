# Copyright (c) 2024, NVIDIA CORPORATION.
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
import networkx as nx
from networkx.classes.digraph import (
    _CachedPropertyResetterAdjAndSucc,
    _CachedPropertyResetterPred,
)
from networkx.classes.graph import (
    _CachedPropertyResetterAdj,
    _CachedPropertyResetterNode,
)

import nx_cugraph as nxcg
from nx_cugraph import _nxver

if _nxver < (3, 4):
    _CACHE_KEY = (True, True, True)
else:
    _CACHE_KEY = (True, True)

# Use to indicate when a full conversion to GPU failed so we don't try again.
_CANT_CONVERT_TO_GPU = "_CANT_CONVERT_TO_GPU"


# `collections.UserDict` was the preferred way to subclass dict, but now
# subclassing dict directly is much better supported and should work here.
# This class should only be necessary if the user clears the cache manually.
class _ZeroGraphCache(dict):
    """Cache that ensures ZeroGraph will reify into a NetworkX graph when cleared."""

    def __init__(self, zero_graph):
        self._zero_graph = zero_graph

    def clear(self):
        self._zero_graph._reify_networkx()
        super().clear()


# TODO: experiment whether we should make `__class__` and other attrs match networkx
class ZeroGraph(nx.Graph):
    __networkx_backend__ = "cugraph"

    _nx_attrs = ("_node", "_adj")

    @property
    def _node(self):
        if (node := self.__dict__["_node"]) is None:
            self._reify_networkx()
            node = self.__dict__["_node"]
        return node

    @_node.setter
    def _node(self, val):
        _CachedPropertyResetterNode.__set__(None, self, val)

    @property
    def _adj(self):
        if (adj := self.__dict__["_adj"]) is None:
            self._reify_networkx()
            adj = self.__dict__["_adj"]
        return adj

    @_adj.setter
    def _adj(self, val):
        _CachedPropertyResetterAdj.__set__(None, self, val)

    @property
    def _is_on_gpu(self):
        cache = getattr(self, "__networkx_cache__", None)
        if not cache:
            return False
        return _CACHE_KEY in cache.get("backends", {}).get("cugraph", {})

    @property
    def _is_on_cpu(self):
        return self.__dict__["_node"] is not None

    @property
    def _cugraph(self):
        nx_cache = getattr(self, "__networkx_cache__", None)
        if nx_cache is None:
            nx_cache = {}
        elif _CANT_CONVERT_TO_GPU in nx_cache:
            return None
        cache = nx_cache.setdefault("backends", {}).setdefault("cugraph", {})
        if (Gcg := cache.get(_CACHE_KEY)) is not None:
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

    @_cugraph.setter
    def _cugraph(self, val):
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
            for key in self._nx_attrs:
                self.__dict__[key] = None

    @nx.Graph.name.setter
    def name(self, s):
        # Don't clear the cache when setting the name, since `.graph` is shared.
        # There is a very small risk here for the cache to become (slightly)
        # insconsistent if graphs from other backends are cached.
        self.graph["name"] = s

    @classmethod
    def to_networkx_class(cls):
        return nx.Graph

    @classmethod
    def to_cugraph_class(cls):
        return nxcg.Graph

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.__networkx_cache__ = _ZeroGraphCache(self)

    def _reify_networkx(self):
        if self.__dict__["_node"] is None:
            # After we make this into an nx graph, we rely on the cache being correct
            Gcg = self._cugraph
            G = nxcg.to_networkx(Gcg)
            for key in self._nx_attrs:
                self.__dict__[key] = G.__dict__[key]


class ZeroDiGraph(nx.DiGraph, ZeroGraph):
    _nx_attrs = ("_node", "_adj", "_succ", "_pred")

    name = ZeroGraph.name
    _node = ZeroGraph._node

    @property
    def _adj(self):
        if (adj := self.__dict__["_adj"]) is None:
            self._reify_networkx()
            adj = self.__dict__["_adj"]
        return adj

    @_adj.setter
    def _adj(self, val):
        _CachedPropertyResetterAdjAndSucc.__set__(None, self, val)

    @property
    def _succ(self):
        if (succ := self.__dict__["_succ"]) is None:
            self._reify_networkx()
            succ = self.__dict__["_succ"]
        return succ

    @_succ.setter
    def _succ(self, val):
        _CachedPropertyResetterAdjAndSucc.__set__(None, self, val)

    @property
    def _pred(self):
        if (pred := self.__dict__["_pred"]) is None:
            self._reify_networkx()
            pred = self.__dict__["_pred"]
        return pred

    @_pred.setter
    def _pred(self, val):
        _CachedPropertyResetterPred.__set__(None, self, val)

    @classmethod
    def to_networkx_class(cls):
        return nx.DiGraph

    @classmethod
    def to_cugraph_class(cls):
        return nxcg.DiGraph


class ZeroMultiGraph(nx.MultiGraph, ZeroGraph):
    name = ZeroGraph.name
    _node = ZeroGraph._node
    _adj = ZeroGraph._adj

    @classmethod
    def to_networkx_class(cls):
        return nx.MultiGraph

    @classmethod
    def to_cugraph_class(cls):
        return nxcg.MultiGraph

    def __init__(self, incoming_graph_data=None, multigraph_input=None, **attr):
        super().__init__(incoming_graph_data, multigraph_input, **attr)
        self.__networkx_cache__ = _ZeroGraphCache(self)


class ZeroMultiDiGraph(nx.MultiDiGraph, ZeroMultiGraph, ZeroDiGraph):
    name = ZeroGraph.name
    _node = ZeroGraph._node
    _adj = ZeroDiGraph._adj
    _succ = ZeroDiGraph._succ
    _pred = ZeroDiGraph._pred

    @classmethod
    def to_networkx_class(cls):
        return nx.MultiDiGraph

    @classmethod
    def to_cugraph_class(cls):
        return nxcg.MultiDiGraph
