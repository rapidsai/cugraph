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

from copy import deepcopy
from typing import TYPE_CHECKING

import cupy as cp
import networkx as nx
import numpy as np
from networkx.classes.digraph import (
    _CachedPropertyResetterAdjAndSucc,
    _CachedPropertyResetterPred,
)

import nx_cugraph as nxcg

from ..utils import index_dtype
from .graph import CudaGraph, Graph

if TYPE_CHECKING:  # pragma: no cover
    from nx_cugraph.typing import AttrKey

__all__ = ["CudaDiGraph", "DiGraph"]

networkx_api = nxcg.utils.decorators.networkx_class(nx.DiGraph)


class DiGraph(nx.DiGraph, Graph):
    _nx_attrs = ("_node", "_adj", "_succ", "_pred")

    name = Graph.name
    _node = Graph._node

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
        _CachedPropertyResetterAdjAndSucc.__set__(None, self, val)
        if cache := getattr(self, "__networkx_cache__", None):
            cache.clear()

    @property
    @networkx_api
    def _succ(self):
        if (succ := self.__dict__["_succ"]) is None:
            self._reify_networkx()
            succ = self.__dict__["_succ"]
        return succ

    @_succ.setter
    def _succ(self, val):
        self._prepare_setter()
        _CachedPropertyResetterAdjAndSucc.__set__(None, self, val)
        if cache := getattr(self, "__networkx_cache__", None):
            cache.clear()

    @property
    @networkx_api
    def _pred(self):
        if (pred := self.__dict__["_pred"]) is None:
            self._reify_networkx()
            pred = self.__dict__["_pred"]
        return pred

    @_pred.setter
    def _pred(self, val):
        self._prepare_setter()
        _CachedPropertyResetterPred.__set__(None, self, val)
        if cache := getattr(self, "__networkx_cache__", None):
            cache.clear()

    @classmethod
    @networkx_api
    def is_directed(cls) -> bool:
        return True

    @classmethod
    @networkx_api
    def is_multigraph(cls) -> bool:
        return False

    @classmethod
    def to_cudagraph_class(cls) -> type[CudaDiGraph]:
        return CudaDiGraph

    @classmethod
    def to_networkx_class(cls) -> type[nx.DiGraph]:
        return nx.DiGraph


class CudaDiGraph(CudaGraph):
    #################
    # Class methods #
    #################

    is_directed = classmethod(DiGraph.is_directed.__func__)
    is_multigraph = classmethod(DiGraph.is_multigraph.__func__)
    to_cudagraph_class = classmethod(DiGraph.to_cudagraph_class.__func__)
    to_networkx_class = classmethod(DiGraph.to_networkx_class.__func__)

    @classmethod
    def _to_compat_graph_class(cls) -> type[DiGraph]:
        return DiGraph

    @networkx_api
    def size(self, weight: AttrKey | None = None) -> int:
        if weight is not None:
            raise NotImplementedError
        return self.src_indices.size

    ##########################
    # NetworkX graph methods #
    ##########################

    @networkx_api
    def reverse(self, copy: bool = True) -> CudaDiGraph:
        return self._copy(not copy, self.__class__, reverse=True)

    @networkx_api
    def to_undirected(self, reciprocal=False, as_view=False):
        N = self._N
        # Upcast to int64 so indices don't overflow
        src_dst_indices_old = N * self.src_indices.astype(np.int64) + self.dst_indices
        if reciprocal:
            src_dst_indices_new = cp.intersect1d(
                src_dst_indices_old,
                self.src_indices + N * self.dst_indices.astype(np.int64),
                # assume_unique=True,  # cupy <= 12.2.0 also assumes sorted
            )
            if self.edge_values:
                sorter = cp.argsort(src_dst_indices_old)
                idx = cp.searchsorted(
                    src_dst_indices_old, src_dst_indices_new, sorter=sorter
                )
                indices = sorter[idx]
                src_indices = self.src_indices[indices].copy()
                dst_indices = self.dst_indices[indices].copy()
                edge_values = {
                    key: val[indices].copy() for key, val in self.edge_values.items()
                }
                edge_masks = {
                    key: val[indices].copy() for key, val in self.edge_masks.items()
                }
            else:
                src_indices, dst_indices = cp.divmod(src_dst_indices_new, N)
                src_indices = src_indices.astype(index_dtype)
                dst_indices = dst_indices.astype(index_dtype)
        else:
            src_dst_indices_old_T = self.src_indices + N * self.dst_indices.astype(
                np.int64
            )
            if self.edge_values:
                src_dst_extra = cp.setdiff1d(
                    src_dst_indices_old_T, src_dst_indices_old, assume_unique=True
                )
                sorter = cp.argsort(src_dst_indices_old_T)
                idx = cp.searchsorted(
                    src_dst_indices_old_T, src_dst_extra, sorter=sorter
                )
                indices = sorter[idx]
                src_indices = cp.hstack((self.src_indices, self.dst_indices[indices]))
                dst_indices = cp.hstack((self.dst_indices, self.src_indices[indices]))
                edge_values = {
                    key: cp.hstack((val, val[indices]))
                    for key, val in self.edge_values.items()
                }
                edge_masks = {
                    key: cp.hstack((val, val[indices]))
                    for key, val in self.edge_masks.items()
                }
            else:
                src_dst_indices_new = cp.union1d(
                    src_dst_indices_old, src_dst_indices_old_T
                )
                src_indices, dst_indices = cp.divmod(src_dst_indices_new, N)
                src_indices = src_indices.astype(index_dtype)
                dst_indices = dst_indices.astype(index_dtype)

        if self.edge_values:
            recip_indices = cp.lexsort(cp.vstack((src_indices, dst_indices)))
            for key, mask in edge_masks.items():
                # Make sure we choose a value that isn't masked out
                val = edge_values[key]
                rmask = mask[recip_indices]
                recip_only = rmask & ~mask
                val[recip_only] = val[recip_indices[recip_only]]
                only = mask & ~rmask
                val[recip_indices[only]] = val[only]
                mask |= mask[recip_indices]
            # Arbitrarily choose to use value from (j > i) edge
            mask = src_indices < dst_indices
            left_idx = cp.nonzero(mask)[0]
            right_idx = recip_indices[mask]
            for val in edge_values.values():
                val[left_idx] = val[right_idx]
        else:
            edge_values = {}
            edge_masks = {}

        node_values = self.node_values
        node_masks = self.node_masks
        key_to_id = self.key_to_id
        id_to_key = None if key_to_id is None else self._id_to_key
        if not as_view:
            node_values = {key: val.copy() for key, val in node_values.items()}
            node_masks = {key: val.copy() for key, val in node_masks.items()}
            if key_to_id is not None:
                key_to_id = key_to_id.copy()
                if id_to_key is not None:
                    id_to_key = id_to_key.copy()
        rv = self.to_undirected_class().from_coo(
            N,
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
        return rv

    # Many more methods to implement...

    ###################
    # Private methods #
    ###################

    def _in_degrees_array(self, *, ignore_selfloops=False):
        dst_indices = self.dst_indices
        if ignore_selfloops:
            not_selfloops = self.src_indices != dst_indices
            dst_indices = dst_indices[not_selfloops]
        if dst_indices.size == 0:
            return cp.zeros(self._N, dtype=np.int64)
        return cp.bincount(dst_indices, minlength=self._N)

    def _out_degrees_array(self, *, ignore_selfloops=False):
        src_indices = self.src_indices
        if ignore_selfloops:
            not_selfloops = src_indices != self.dst_indices
            src_indices = src_indices[not_selfloops]
        if src_indices.size == 0:
            return cp.zeros(self._N, dtype=np.int64)
        return cp.bincount(src_indices, minlength=self._N)
