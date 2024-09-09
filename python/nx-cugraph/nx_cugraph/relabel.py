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
import itertools
from collections import defaultdict

import cupy as cp
import networkx as nx
import numpy as np

import nx_cugraph as nxcg

from .utils import _get_int_dtype, _groupby, index_dtype, networkx_algorithm

__all__ = [
    "convert_node_labels_to_integers",
    "relabel_nodes",
]


@networkx_algorithm(version_added="24.08")
def relabel_nodes(G, mapping, copy=True):
    G_orig = G
    if isinstance(G, nx.Graph):
        is_compat_graph = isinstance(G, nxcg.Graph)
        if not copy and not is_compat_graph:
            raise RuntimeError(
                "Using `copy=False` is invalid when using a NetworkX graph "
                "as input to `nx_cugraph.relabel_nodes`"
            )
        G = nxcg.from_networkx(G, preserve_all_attrs=True)
    else:
        is_compat_graph = False

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
    key_to_newid = dict(zip(newid_to_key, range(len(newid_to_key))))

    src_indices = G.src_indices
    dst_indices = G.dst_indices
    edge_values = G.edge_values
    edge_masks = G.edge_masks
    node_values = G.node_values
    node_masks = G.node_masks
    if G.is_multigraph():
        edge_indices = G.edge_indices
        edge_keys = G.edge_keys
    if len(key_to_previd) != G._N:
        # Some nodes were combined.
        # Node data doesn't get merged, so use the data from the last shared index
        int_dtype = _get_int_dtype(G._N - 1)
        node_indices = cp.fromiter(key_to_previd.values(), int_dtype)
        node_indices_np = node_indices.get()  # Node data may be cupy or numpy arrays
        node_values = {key: val[node_indices_np] for key, val in node_values.items()}
        node_masks = {key: val[node_indices_np] for key, val in node_masks.items()}

        # Renumber, but will have duplicates
        translations = cp.fromiter(
            (key_to_newid[key] for key in previd_to_key), index_dtype
        )
        src_indices_dup = translations[src_indices]
        dst_indices_dup = translations[dst_indices]

        if G.is_multigraph():
            # No merging necessary for multigraphs.
            if G.is_directed():
                src_indices = src_indices_dup
                dst_indices = dst_indices_dup
            else:
                # New self-edges should have one edge entry, not two
                mask = (
                    # Not self-edges, no need to deduplicate
                    (src_indices_dup != dst_indices_dup)
                    # == : already self-edges; no need to deduplicate
                    # < : if new self-edges, keep where src < dst
                    | (src_indices <= dst_indices)
                )
                if mask.all():
                    src_indices = src_indices_dup
                    dst_indices = dst_indices_dup
                else:
                    src_indices = src_indices_dup[mask]
                    dst_indices = dst_indices_dup[mask]
                    if edge_values:
                        edge_values = {
                            key: val[mask] for key, val in edge_values.items()
                        }
                        edge_masks = {key: val[mask] for key, val in edge_masks.items()}
                    if edge_keys is not None:
                        edge_keys = [
                            key for keep, key in zip(mask.tolist(), edge_keys) if keep
                        ]
                    if edge_indices is not None:
                        edge_indices = edge_indices[mask]
            # Handling of `edge_keys` and `edge_indices` is pure Python to match nx.
            # This may be slower than we'd like; if it's way too slow, should we
            # direct users to use the defaults of None?
            if edge_keys is not None:
                seen = set()
                new_edge_keys = []
                for key in zip(src_indices.tolist(), dst_indices.tolist(), edge_keys):
                    if key in seen:
                        src, dst, edge_key = key
                        if not isinstance(edge_key, (int, float)):
                            edge_key = 0
                        for edge_key in itertools.count(edge_key):
                            if (src, dst, edge_key) not in seen:
                                seen.add((src, dst, edge_key))
                                break
                    else:
                        seen.add(key)
                        edge_key = key[2]
                    new_edge_keys.append(edge_key)
                edge_keys = new_edge_keys
            if edge_indices is not None:
                # PERF: can we do this using cupy?
                seen = set()
                new_edge_indices = []
                for key in zip(
                    src_indices.tolist(), dst_indices.tolist(), edge_indices.tolist()
                ):
                    if key in seen:
                        src, dst, edge_index = key
                        for edge_index in itertools.count(edge_index):
                            if (src, dst, edge_index) not in seen:
                                seen.add((src, dst, edge_index))
                                break
                    else:
                        seen.add(key)
                        edge_index = key[2]
                    new_edge_indices.append(edge_index)
                edge_indices = cp.array(new_edge_indices, index_dtype)
        else:
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
                mask = cp.ones(src_indices.size, dtype=bool)
                mask[ind] = False
                edge_data = [val[mask] for val in edge_values.values()]
                edge_data.extend(val[mask] for val in edge_masks.values())
                groups = _groupby(inv[mask], edge_data)

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

                int_dtype = _get_int_dtype(src_indices.size - 1)
                for k, v in values_to_update.items():
                    ii, jj = zip(*v)
                    edge_val = edge_values[k]
                    edge_val[cp.array(ii, dtype=int_dtype)] = cp.array(
                        jj, dtype=edge_val.dtype
                    )
                for k, v in masks_to_update.items():
                    ii, jj = zip(*v)
                    edge_masks[k][cp.array(ii, dtype=int_dtype)] = cp.array(
                        jj, dtype=bool
                    )
            src_indices = stacked[0]
            dst_indices = stacked[1]

    if G.is_multigraph():
        # `edge_keys` and `edge_indices` are preserved for free if no nodes were merged
        extra_kwargs = {"edge_keys": edge_keys, "edge_indices": edge_indices}
    else:
        extra_kwargs = {}
    rv = G.__class__.from_coo(
        len(key_to_previd),
        src_indices,
        dst_indices,
        edge_values=edge_values,
        edge_masks=edge_masks,
        node_values=node_values,
        node_masks=node_masks,
        id_to_key=newid_to_key,
        key_to_id=key_to_newid,
        use_compat_graph=is_compat_graph,
        **extra_kwargs,
    )
    rv.graph.update(G.graph)
    if not copy:
        G_orig._become(rv)
        return G_orig
    return rv


@networkx_algorithm(version_added="24.08")
def convert_node_labels_to_integers(
    G, first_label=0, ordering="default", label_attribute=None
):
    if ordering not in {"default", "sorted", "increasing degree", "decreasing degree"}:
        raise nx.NetworkXError(f"Unknown node ordering: {ordering}")
    if isinstance(G, nx.Graph):
        is_compat_graph = isinstance(G, nxcg.Graph)
        G = nxcg.from_networkx(G, preserve_all_attrs=True)
    else:
        is_compat_graph = False
    G = G.copy()
    if label_attribute is not None:
        prev_vals = G.id_to_key
        if prev_vals is None:
            prev_vals = cp.arange(G._N, dtype=_get_int_dtype(G._N - 1))
        else:
            try:
                prev_vals = np.array(prev_vals)
            except ValueError:
                prev_vals = np.fromiter(prev_vals, object)
            else:
                try:
                    prev_vals = cp.array(prev_vals)
                except ValueError:
                    pass
        G.node_values[label_attribute] = prev_vals
        G.node_masks.pop(label_attribute, None)
    id_to_key = None
    if ordering == "default" or ordering == "sorted" and G.key_to_id is None:
        if first_label == 0:
            G.key_to_id = None
        else:
            id_to_key = list(range(first_label, first_label + G._N))
            G.key_to_id = dict(zip(id_to_key, range(G._N)))
    elif ordering == "sorted":
        key_to_id = G.key_to_id
        G.key_to_id = {
            i: key_to_id[n] for i, n in enumerate(sorted(key_to_id), first_label)
        }
    else:
        pairs = sorted(
            ((d, n) for (n, d) in G._nodearray_to_dict(G._degrees_array()).items()),
            reverse=ordering == "decreasing degree",
        )
        key_to_id = G.key_to_id
        G.key_to_id = {i: key_to_id[n] for i, (d, n) in enumerate(pairs, first_label)}
    G._id_to_key = id_to_key
    if is_compat_graph:
        return G._to_compat_graph()
    return G
