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

import nx_cugraph as nxcg

from ..utils import index_dtype
from .graph import Graph

if TYPE_CHECKING:  # pragma: no cover
    from nx_cugraph.typing import NodeKey

__all__ = ["DiGraph"]

networkx_api = nxcg.utils.decorators.networkx_class(nx.DiGraph)


class DiGraph(Graph):
    #################
    # Class methods #
    #################

    @classmethod
    @networkx_api
    def is_directed(cls) -> bool:
        return True

    @classmethod
    def to_networkx_class(cls) -> type[nx.DiGraph]:
        return nx.DiGraph

    @networkx_api
    def number_of_edges(
        self, u: NodeKey | None = None, v: NodeKey | None = None
    ) -> int:
        if u is not None or v is not None:
            raise NotImplementedError
        return self.src_indices.size

    ##########################
    # NetworkX graph methods #
    ##########################

    @networkx_api
    def reverse(self, copy: bool = True) -> DiGraph:
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
                src_indices, dst_indices = cp.divmod(
                    src_dst_indices_new, N, dtype=index_dtype
                )
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
                src_indices, dst_indices = cp.divmod(
                    src_dst_indices_new, N, dtype=index_dtype
                )

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

    def _in_degrees_array(self):
        return cp.bincount(self.dst_indices, minlength=self._N)

    def _out_degrees_array(self):
        return cp.bincount(self.src_indices, minlength=self._N)
