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
from collections import defaultdict

import cupy as cp
import networkx as nx

import nx_cugraph as nxcg

from .utils import _get_int_dtype, _groupby, index_dtype, networkx_algorithm

__all__ = [
    "relabel_nodes",
]


@networkx_algorithm(version_added="24.08")
def relabel_nodes(G, mapping, copy=True):
    if isinstance(G, nx.Graph):
        G = nxcg.from_networkx(G, preserve_all_attrs=True)
    it = range(G._N) if G.key_to_id is None else G.id_to_key
    if callable(mapping):
        previd_to_key = [mapping(node) for node in it]
    else:
        previd_to_key = [mapping.get(node, node) for node in it]
    if not copy:
        # Our implementation does not need to raise here, but do so to match networkx.
        it = range(G._N) if G.key_to_id is None else G.id_to_key
        D = nx.DiGraph([(x, y) for x, y in zip(it, previd_to_key) if x != y])
        if nx.algorithms.dag.has_cycle(D):
            raise nx.NetworkXUnfeasible(
                "The node label sets are overlapping and no ordering can "
                "resolve the mapping. Use copy=True."
            )
    key_to_previd = {val: i for i, val in enumerate(previd_to_key)}
    newid_to_key = list(key_to_previd)

    src_indices = G.src_indices
    dst_indices = G.dst_indices
    edge_values = G.edge_values
    edge_masks = G.edge_masks
    node_values = G.node_values
    node_masks = G.node_masks
    if len(key_to_previd) != G._N:  # Some nodes were combined
        # Node data doesn't get merged, so use the data from the last shared index
        int_dtype = _get_int_dtype(G._N)
        node_indices = cp.fromiter(key_to_previd.values(), int_dtype)
        node_indices_np = node_indices.get()  # Node data may be cupy or numpy arrays
        node_values = {key: val[node_indices_np] for key, val in node_values.items()}
        node_masks = {key: val[node_indices_np] for key, val in node_masks.items()}

        # Renumber, but will have duplicates
        src_indices_dup = cp.searchsorted(node_indices, src_indices).astype(index_dtype)
        dst_indices_dup = cp.searchsorted(node_indices, dst_indices).astype(index_dtype)
        stacked_dup = cp.vstack((src_indices_dup, dst_indices_dup))
        if not edge_values:
            # Drop duplicates
            stacked = cp.unique(stacked_dup, axis=1)
        else:
            # Drop duplicates. This relies heavily on `_groupby`.
            # It has not been compared to alternative implementations.
            # I wonder if there are ways to use assignment using duplicate indices.
            (stacked, ind, inv) = cp.unique(
                stacked_dup, axis=1, return_index=True, return_inverse=True
            )
            if ind.dtype != int_dtype:
                ind = ind.astype(int_dtype)
            if inv.dtype != int_dtype:
                inv = inv.astype(int_dtype)

            # We need to merge edge data
            dup_ids = cp.setdiff1d(cp.arange(src_indices.size, dtype=int_dtype), ind)
            edge_data = [val[dup_ids] for val in edge_values.values()]
            edge_data.extend(val[dup_ids] for val in edge_masks.values())
            groups = _groupby(inv[dup_ids], edge_data)

            edge_values = {key: val[ind] for key, val in edge_values.items()}
            edge_masks = {key: val[ind] for key, val in edge_masks.items()}

            value_keys = list(edge_values.keys())
            mask_keys = list(edge_masks.keys())

            values_to_update = defaultdict(list)
            masks_to_update = defaultdict(list)
            for k, v in groups.items():
                it = iter(v)
                vals = dict(zip(value_keys, it))  # zip(strict=False)
                masks = dict(zip(mask_keys, it))  # zip(strict=True)
                for key, val in vals.items():
                    if key in masks:
                        val = val[masks[key]]
                        if val.size > 0:
                            values_to_update[key].append((k, val[-1]))
                            masks_to_update[key].append((k, True))
                    else:
                        values_to_update[key].append((k, val[-1]))
                        if key in edge_masks:
                            masks_to_update[key].append((k, True))

            int_dtype = _get_int_dtype(src_indices.size)
            for k, v in values_to_update.items():
                ii, jj = zip(*v)
                edge_val = edge_values[k]
                edge_val[cp.array(ii, dtype=int_dtype)] = cp.array(
                    jj, dtype=edge_val.dtype
                )
            for k, v in masks_to_update.items():
                ii, jj = zip(*v)
                edge_masks[k][cp.array(ii, dtype=int_dtype)] = cp.array(jj, dtype=bool)
        src_indices = stacked[0]
        dst_indices = stacked[1]

    rv = G.__class__.from_coo(
        len(key_to_previd),
        src_indices,
        dst_indices,
        edge_values=edge_values,
        edge_masks=edge_masks,
        node_values=node_values,
        node_masks=node_masks,
        id_to_key=newid_to_key,
    )
    rv.graph.update(G.graph)
    if not copy:
        G._become(rv)
        return G
    return rv


@relabel_nodes._can_run
def _(G, mapping, copy=True):
    return not G.is_multigraph()  # TODO
