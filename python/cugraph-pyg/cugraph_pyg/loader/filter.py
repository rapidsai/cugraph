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

import cupy

from cugraph_pyg.data import CuGraphStore

from typing import (
    Dict,
    Sequence,
)


def _filter_cugraph_store(
    feature_store: CuGraphStore,
    graph_store: CuGraphStore,
    node_dict: Dict[str, Sequence],
    row_dict: Dict[str, Sequence],
    col_dict: Dict[str, Sequence],
    edge_dict: Dict[str, Sequence],
) -> dict:
    """
    Primarily for testing without torch and torch_geometric.
    Returns a dictionary containing the sampled subgraph.
    """
    data = {}

    for attr in graph_store.get_all_edge_attrs():
        key = attr.edge_type
        if key in row_dict and key in col_dict:
            edge_index = cupy.stack([row_dict[key], col_dict[key]])
            data[attr.edge_type] = {}
            data[attr.edge_type]["edge_index"] = edge_index

    # Filter node storage:
    required_attrs = []
    for attr in feature_store.get_all_tensor_attrs():
        if attr.group_name in node_dict:
            attr.index = node_dict[attr.group_name]
            required_attrs.append(attr)
            data[attr.group_name] = {}
            data["num_nodes"] = attr.index.size
    tensors = feature_store.multi_get_tensor(required_attrs)
    for i, attr in enumerate(required_attrs):
        data[attr.group_name][attr.attr_name] = tensors[i]

    return data
