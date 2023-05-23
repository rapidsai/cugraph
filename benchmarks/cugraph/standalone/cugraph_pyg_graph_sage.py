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


import json
import time
import argparse
import gc
import os

import torch
import numpy as np

from torch_geometric.nn import CuGraphSAGEConv

import torch.nn as nn
import torch.nn.functional as F

from typing import Union

def load_disk_features(meta, node_type, replication_factor=1):
    node_type_path = os.path.join(meta['dataset_dir'], meta['dataset'], 'npy', node_type)
    
    if replication_factor == 1:
        return np.load(
            os.path.join(node_type_path, 'node_feat.npy'),
            mmap_mode='r'
        )

    else:
        return np.load(
            os.path.join(node_type_path, f'node_feat_{replication_factor}x.npy'),
            mmap_mode='r'
        )


class CuGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(CuGraphSAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            conv = CuGraphSAGEConv(hidden_channels, hidden_channels)
            self.convs.append(conv)

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge, size):
        edge_csc = CuGraphSAGEConv.to_csc(edge, (size[0], size[0]))
        for conv in self.convs:
            x = conv(x, edge_csc)[: size[1]]
            x = F.relu(x)
            x = F.dropout(x, p=0.5)

        return self.lin(x)


def init_pytorch_worker(device_id: int) -> None:
    import cupy
    import rmm

    rmm.reinitialize(
        devices=[device_id],
        pool_allocator=True,
    )

    cupy.cuda.Device(device_id).use()
    torch.cuda.set_device(device_id)

    from rmm.allocators.torch import rmm_torch_allocator
    torch.cuda.change_current_allocator(rmm_torch_allocator)

    from rmm.allocators.cupy import rmm_cupy_allocator
    cupy.cuda.set_allocator(rmm_cupy_allocator)


def train_native(device:int, features_device:Union[str, int] = "cpu", num_epochs=1) -> None:
    pass

def train(device: int, meta: dict, features_device: Union[str, int] = "cpu", num_epochs=1) -> None:
    """
    Parameters
    ----------
    device: int
        The CUDA device where the model, graph data, and node labels will be stored.
    features_device: Union[str, int]
        The device (CUDA device or CPU) where features will be stored.
    """

    init_pytorch_worker(device)

    import cudf
    import cugraph
    from cugraph_pyg.data import CuGraphStore
    from cugraph_pyg.loader import CuGraphNeighborLoader

    dataset_path = os.path.join(meta['dataset_dir'], meta['dataset'])
    with open(os.path.join(dataset_path, 'meta.json'), 'r') as f:
        input_meta = json.load(f)

    replication_factor = meta['replication_factor']
    G = {tuple(edge_type.split('__')): t * replication_factor for edge_type, t in input_meta['num_edges'].items()}
    N = {node_type: t * replication_factor for node_type, t in input_meta['num_nodes'].items()}

    fs = cugraph.gnn.FeatureStore(backend="torch")

    for node_type in os.listdir(dataset_path, 'npy'):
        feature_data = load_disk_features(meta, node_type, replication_factor=replication_factor)
        fs.add_data(
            torch.as_tensor(feature_data, device=features_device),
            node_type,
            "x",
        )

        label_path = os.path.join(dataset_path, 'parquet', node_type, 'node_label.parquet')
        if os.path.exists(label_path):
            node_label = cudf.read_parquet(label_path)
            node_label_tensor = torch.full(N[node_type], -1, dtype=torch.float32)
            node_label_tensor[torch.as_tensor(node_label.node.values, device='cuda')] = \
                torch.as_tensor(node_label.label.values, device='cuda')
            
            del node_label
            gc.collect()

            fs.add_data(torch.as_tensor(node_label, device=device), node_type, 'y')

    cugraph_store = CuGraphStore(fs, G, N)

    model = (
        CuGraphSAGE(in_channels=128, hidden_channels=64, out_channels=349, num_layers=3)
        .to(torch.float32)
        .to(device)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        start_time_train = time.perf_counter_ns()
        model.train()

        cugraph_bulk_loader = CuGraphNeighborLoader(
            cugraph_store, train_nodes, batch_size=500, num_neighbors=[10, 25]
        )

        total_loss = 0
        num_batches = 0

        # This context manager will handle different # batches per rank
        # barrier() cannot do this since the number of ops per rank is
        # different.  It essentially acts like barrier would if the
        # number of ops per rank was the same.
        for epoch in range(num_epochs):
            for iter_i, hetero_data in enumerate(cugraph_bulk_loader):
                num_batches += 1
                if iter_i % 20 == 0:
                    print(f"iteration {iter_i}")

                # train
                train_mask = hetero_data.train_dict["paper"]
                y_true = hetero_data.y_dict["paper"]

                y_pred = model(
                    hetero_data.x_dict["paper"].to(device).to(torch.float32),
                    hetero_data.edge_index_dict[("paper", "cites", "paper")].to(device),
                    (len(y_true), len(y_true)),
                )

                y_true = F.one_hot(
                    y_true[train_mask].to(torch.int64), num_classes=349
                ).to(torch.float32)

                y_pred = y_pred[train_mask]

                loss = F.cross_entropy(y_pred, y_true)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                del y_true
                del y_pred
                del loss
                del hetero_data
                gc.collect()

            end_time_train = time.perf_counter_ns()
            print(
                f"epoch {epoch} time: "
                f"{(end_time_train - start_time_train) / 1e9:3.4f} s"
            )
            print(f"loss after epoch {epoch}: {total_loss / num_batches}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU to allocate to pytorch for model, graph data, and node label storage",
        required=False,
    )

    parser.add_argument(
        "--features_device",
        type=str,
        default="0",
        help="Device to allocate to pytorch for feature storage",
        required=False,
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
        required=False,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        features_device = int(args.features_device)
    except ValueError:
        features_device = args.features_device

    train(args.device, features_device, args.num_epochs)


if __name__ == "__main__":
    main()
