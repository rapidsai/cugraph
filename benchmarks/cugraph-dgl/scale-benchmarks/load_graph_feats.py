# Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

import numpy as np
import pandas as pd
import torch
import os


def load_edges_from_disk(parquet_path, replication_factor, input_meta):
    """
    Load the edges from disk into a graph data dictionary.
    Args:
        parquet_path: Path to the parquet directory.
        replication_factor: Number of times to replicate the edges.
        input_meta: Input meta data.
    Returns:
        dict: Dictionary of edge types to a tuple of (src, dst)
    """
    graph_data = {}

    for edge_type in input_meta["num_edges"].keys():
        print(f"Loading edge index for edge type {edge_type} for replication factor = {replication_factor}")

        canonical_edge_type = tuple(edge_type.split("__"))
        edge_index = pd.read_parquet(os.path.join(parquet_path, edge_type, "edge_index.parquet"))
        edge_index = {
            "src": torch.from_numpy(edge_index.src.values),
            "dst": torch.from_numpy(edge_index.dst.values),
        }

        if replication_factor > 1:
            src_list, dst_list = replicate_edges(edge_index, canonical_edge_type, replication_factor, input_meta)
            edge_index["src"] = torch.cat(src_list).contiguous()
            edge_index["dst"] = torch.cat(dst_list).contiguous()

        graph_data[canonical_edge_type] = edge_index["src"], edge_index["dst"]

    print("Read Edge Data")
    return graph_data


def replicate_edges(edge_index, canonical_edge_type, replication_factor, input_meta):
    src_list = [edge_index["src"]]
    dst_list = [edge_index["dst"]]

    for r in range(1, replication_factor):
        new_src = edge_index["src"] + (r * input_meta["num_nodes"][canonical_edge_type[0]])
        new_dst = edge_index["dst"] + (r * input_meta["num_nodes"][canonical_edge_type[2]])
        src_list.append(new_src)
        dst_list.append(new_dst)

    return src_list, dst_list




def load_node_labels(dataset_path, replication_factor, input_meta):
    num_nodes_dict = {node_type: t * replication_factor for node_type, t in input_meta["num_nodes"].items()}
    node_data = {}

    for node_type in input_meta["num_nodes"].keys():
        node_data[node_type] = {}
        label_path = os.path.join(dataset_path, "parquet", node_type, "node_label.parquet")

        if os.path.exists(label_path):
            node_data[node_type] = process_node_label(label_path, node_type, replication_factor, num_nodes_dict, input_meta)

        else:
            node_data[node_type]["num_nodes"] = num_nodes_dict[node_type]

    print("Loaded node labels", flush=True)
    return node_data

def process_node_label(label_path, node_type, replication_factor, num_nodes_dict, input_meta):
    node_label = pd.read_parquet(label_path)

    if replication_factor > 1:
        node_label = replicate_node_label(node_label, node_type, replication_factor, input_meta)

    node_label_tensor = torch.full((num_nodes_dict[node_type],), -1, dtype=torch.float32)
    node_label_tensor[torch.as_tensor(node_label.node.values)] = torch.as_tensor(node_label.label.values)

    del node_label

    return {
        "train_idx": (node_label_tensor > -1).contiguous().nonzero().view(-1),
        "y": node_label_tensor.contiguous().long()
    }


def replicate_node_label(node_label, node_type, replication_factor, input_meta):
    base_num_nodes = input_meta["num_nodes"][node_type]

    replicated_df = pd.DataFrame({
        "node": pd.concat([node_label.node + (r * base_num_nodes) for r in range(1, replication_factor)]),
        "label": pd.concat([node_label.label for _ in range(1, replication_factor)])
    })

    return pd.concat([node_label, replicated_df]).reset_index(drop=True)


def load_node_features(dataset_path, replication_factor, node_type):
    print("Loading node features", flush=True)
    node_type_path = os.path.join(dataset_path, "npy", node_type)
    if replication_factor == 1:
        fname =  os.path.join(node_type_path, "node_feat.npy")
    else:
        fname = os.path.join(node_type_path, f"node_feat_{replication_factor}x.npy")
    
    feat = torch.from_numpy(np.load(fname))
    print("Loaded node features", flush=True)
    return feat
