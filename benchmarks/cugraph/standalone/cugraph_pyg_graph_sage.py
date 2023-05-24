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
from torch_geometric.utils.trim_to_layer import TrimToLayer

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
        self.convs.append(CuGraphSAGEConv(in_channels, hidden_channels, aggr='mean'))
        for _ in range(num_layers - 2):
            conv = CuGraphSAGEConv(hidden_channels, hidden_channels, aggr='mean')
            self.convs.append(conv)
        
        self.convs.append(CuGraphSAGEConv(hidden_channels, out_channels, aggr='mean'))

        self._trim = TrimToLayer()

    def forward(self, x, edge, num_sampled_nodes, num_sampled_edges):
        print(num_sampled_nodes)
        print(num_sampled_edges)

        for i, conv in enumerate(self.convs):
            edge = edge.cuda()
            x = x.cuda().to(torch.float32)

            _, edge, _ = self._trim(
                i,
                num_sampled_nodes,
                num_sampled_edges,
                x,
                edge,
                None
            )
            print(edge.shape)
            print(edge)
            
            s = len(edge.unique())
            edge_csc = CuGraphSAGEConv.to_csc(edge, (s, s))

            print('x shape', x.shape)
            print('s: ', s)
            x = x[:s]

            print(x.shape)
            print(edge.shape)
            print('----------------')

            x = conv(x, edge_csc)
            x = F.relu(x)
            x = F.dropout(x, p=0.5)

        return x


def init_pytorch_worker(device_id: int) -> None:
    import cupy
    import rmm

    rmm.reinitialize(
        devices=[device_id],
        pool_allocator=False,
    )


    from rmm.allocators.torch import rmm_torch_allocator
    torch.cuda.change_current_allocator(rmm_torch_allocator)

    from rmm.allocators.cupy import rmm_cupy_allocator
    cupy.cuda.set_allocator(rmm_cupy_allocator)

    cupy.cuda.Device(device_id).use()
    torch.cuda.set_device(device_id)



def train_native(device:int, features_device:Union[str, int] = "cpu", num_epochs=1) -> None:
    pass

def train(bulk_samples_dir: str, device: int, features_device: Union[str, int] = "cpu", num_epochs=1) -> None:
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
    from cugraph_pyg.loader import BulkSampleLoader

    with open(os.path.join(bulk_samples_dir, 'output_meta.json'), 'r') as f:
        output_meta = json.load(f)

    dataset_path = os.path.join(output_meta['dataset_dir'], output_meta['dataset'])
    with open(os.path.join(dataset_path, 'meta.json'), 'r') as f:
        input_meta = json.load(f)

    replication_factor = output_meta['replication_factor']
    G = {tuple(edge_type.split('__')): t * replication_factor for edge_type, t in input_meta['num_edges'].items()}
    N = {node_type: t * replication_factor for node_type, t in input_meta['num_nodes'].items()}

    fs = cugraph.gnn.FeatureStore(backend="torch")

    num_input_features = 0
    num_output_features = 0
    for node_type in os.listdir(os.path.join(dataset_path, 'npy')):
        feature_data = load_disk_features(output_meta, node_type, replication_factor=replication_factor)
        fs.add_data(
            torch.as_tensor(feature_data, device=features_device),
            node_type,
            "x",
        )
        if feature_data.shape[1] > num_input_features:
            num_input_features = feature_data.shape[1]

        label_path = os.path.join(dataset_path, 'parquet', node_type, 'node_label.parquet')
        if os.path.exists(label_path):
            node_label = cudf.read_parquet(label_path)
            node_label_tensor = torch.full((N[node_type],), -1, dtype=torch.float32, device='cuda')
            node_label_tensor[torch.as_tensor(node_label.node.values, device='cuda')] = \
                torch.as_tensor(node_label.label.values, device='cuda')
            
            del node_label
            gc.collect()

            fs.add_data((node_label_tensor > -1), node_type, 'train')
            fs.add_data(node_label_tensor, node_type, 'y')
            num_classes = int(node_label_tensor.max()) + 1
            if num_classes > num_output_features:
                num_output_features = num_classes
    print('done loading data')

    print(num_input_features, num_output_features, len(output_meta['fanout']))
    
    model = CuGraphSAGE(
            in_channels=num_input_features,
            hidden_channels=64,
            out_channels=num_output_features,
            num_layers=len(output_meta['fanout'])
    ).to(torch.float32).to(device)
    print('done creating model')
    
    cugraph_store = CuGraphStore(fs, G, N)
    print('done creating store')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        start_time_train = time.perf_counter_ns()
        model.train()

        cugraph_loader = BulkSampleLoader(
            cugraph_store,
            cugraph_store,
            input_nodes=None,
            directory=os.path.join(bulk_samples_dir, 'samples'),
        )
        print('done creating loader')

        total_loss = 0
        num_batches = 0

        for epoch in range(num_epochs):
            t = time.perf_counter()
            for iter_i, data in enumerate(cugraph_loader):
                print(time.perf_counter() - t)
                print(len(data.edge_index_dict['paper','cites','paper'][0].unique()))
                print(len(data.edge_index_dict['paper','cites','paper'][1].unique()))
                print('*********************************************************')
                data = data.to_homogeneous()

                num_batches += 1
                if iter_i % 20 == 0:
                    print(f"iteration {iter_i}")

                # train
                y_true = data.y

                y_pred = model(
                    data.x,
                    data.edge_index,
                    data.num_sampled_nodes,
                    data.num_sampled_edges,
                )

                if y_pred.shape[0] > len(y_true):
                    raise ValueError(f"illegal shape: {y_pred.shape}; {y_true.shape}")

                y_true = y_true[:y_pred.shape[0]]

                y_true = F.one_hot(
                    y_true.to(torch.int64), num_classes=y_pred.shape[1]
                ).to(torch.float32)
                print('shape: ', y_true.shape)
                """

                loss = F.cross_entropy(y_pred, y_true)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                

                del y_true
                del y_pred
                del loss
                del data
                gc.collect()
                """
                t = time.perf_counter()

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

    parser.add_argument(
        "--sample_dir",
        type=str,
        help="Directory with stored bulk samples",
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        features_device = int(args.features_device)
    except ValueError:
        features_device = args.features_device

    train(args.sample_dir, device=args.device, features_device=features_device, num_epochs=args.num_epochs)


if __name__ == "__main__":
    main()
