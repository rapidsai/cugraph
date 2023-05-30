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


import re
import json
import time
import argparse
import gc
import os

import torch
import numpy as np

import torch.nn.functional as F

from typing import Union

from models_cugraph import CuGraphSAGE

def load_disk_features(meta: dict, node_type: str, replication_factor: int = 1):
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


def train_epoch(model, loader, optimizer):
    total_loss = 0.0
    num_batches = 0

    t = time.perf_counter()
    for iter_i, data in enumerate(loader):
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
    
    return total_loss, num_batches


def train_native(device:int, features_device:Union[str, int] = "cpu", num_epochs=1) -> None:
    from models_native import GraphSAGE
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import NeighborLoader

    import cudf
    import pandas

    with open(os.path.join(bulk_samples_dir, 'output_meta.json'), 'r') as f:
        output_meta = json.load(f)

    dataset_path = os.path.join(output_meta['dataset_dir'], output_meta['dataset'])
    with open(os.path.join(dataset_path, 'meta.json'), 'r') as f:
        input_meta = json.load(f)

    replication_factor = output_meta['replication_factor']
    
    num_edges_dict = {tuple(edge_type.split('__')): t * replication_factor for edge_type, t in input_meta['num_edges'].items()}
    num_nodes_dict = {node_type: t * replication_factor for node_type, t in input_meta['num_nodes'].items()}

    hetero_data = HeteroData()
    num_input_features = 0
    num_output_features = 0
    for node_type in os.listdir(os.path.join(dataset_path, 'npy')):
        feature_data = load_disk_features(output_meta, node_type, replication_factor=replication_factor)
        hetero_data[node_type].x = feature_data.to(features_device)

        if feature_data.shape[1] > num_input_features:
            num_input_features = feature_data.shape[1]

        label_path = os.path.join(dataset_path, 'parquet', node_type, 'node_label.parquet')
        if os.path.exists(label_path):
            node_label = cudf.read_parquet(label_path)
            node_label_tensor = torch.full((num_nodes_dict[node_type],), -1, dtype=torch.float32, device='cuda')
            node_label_tensor[torch.as_tensor(node_label.node.values, device='cuda')] = \
                torch.as_tensor(node_label.label.values, device='cuda')
            
            del node_label
            gc.collect()

            hetero_data[node_type]['train'] = (node_label_tensor > -1)
            hetero_data[node_type]['y'] = node_label_tensor

            num_classes = int(node_label_tensor.max()) + 1
            if num_classes > num_output_features:
                num_output_features = num_classes

    print('done loading feature data')

    # Have to load graph data for native PyG
    parquet_path = os.path.join(
        output_meta['dataset_dir'],
        output_meta['dataset'],
        'parquet'
    )

    for edge_type in os.listdir(parquet_path):
        if re.match(r'[a-z]+__[a-z]+__[a-z]+', edge_type):
            print(f'Loading edge index for edge type {edge_type}')

            can_edge_type = tuple(edge_type.split('__'))
            ei = pandas.read_parquet(os.path.join(os.path.join(parquet_path, edge_type), 'edge_index.parquet'))
            ei = {
                'src': torch.as_tensor(ei['src']),
                'dst': torch.as_tensor(ei['dst']),
            }
            gc.collect()

            if replication_factor > 1:
                for r in range(1, replication_factor):
                    ei['src'] = torch.concat([
                        ei['src'],
                        ei['src'] + int(r * output_meta['num_nodes'][edge_type[0]]),
                    ])

                    ei['dst'] = torch.concat([
                        ei['dst'],
                        ei['dst'] + int(r * output_meta['num_nodes'][edge_type[2]]),
                    ])
            gc.collect()

            ei = torch.stack([
                ei['src'],
                ei['dst'],
            ])
            gc.collect()

            hetero_data[can_edge_type]['edge_index'] = ei
            gc.collect()

    print('done loading graph data')    
    print(num_input_features, num_output_features, len(output_meta['fanout']))
    
    model = GraphSAGE(
            in_channels=num_input_features,
            hidden_channels=64,
            out_channels=num_output_features,
            num_layers=len(output_meta['fanout'])
    ).to(torch.float32).to(device)
    print('done creating model')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        start_time_train = time.perf_counter_ns()
        model.train()
        
        loader = NeighborLoader(
            hetero_data,
            hetero_data,
            input_nodes=None,
        )
        print('done creating loader')

        total_loss, num_batches = train_epoch(model, loader, optimizer)

        end_time_train = time.perf_counter_ns()
        print(
            f"epoch {epoch} time: "
            f"{(end_time_train - start_time_train) / 1e9:3.4f} s"
        )
        print(f"loss after epoch {epoch}: {total_loss / num_batches}")

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

        total_loss, num_batches = train_epoch(model, cugraph_loader, optimizer)

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
