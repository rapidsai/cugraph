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
import socket

import torch
import numpy as np
import pandas

import torch.nn.functional as F
import torch.distributed as td
import torch.multiprocessing as tmp
from torch.nn.parallel import DistributedDataParallel as ddp
from torch.distributed.optim import ZeroRedundancyOptimizer

from typing import Union, List

from models_cugraph import CuGraphSAGE
from cugraph.testing.mg_utils import enable_spilling

disk_features = {}

def load_disk_features(meta: dict, node_type: str, replication_factor: int = 1):
    node_type_path = os.path.join(meta['dataset_dir'], meta['dataset'], 'npy', node_type)
    
    if replication_factor == 1:
        full_path = os.path.join(node_type_path, 'node_feat.npy')
        if full_path in disk_features:
            return disk_features[full_path]
        disk_features[full_path] = np.load(
            full_path,
            mmap_mode='r'
        )
        return disk_features[full_path]

    else:
        full_path = os.path.join(node_type_path, f'node_feat_{replication_factor}x.npy')
        if full_path in disk_features:
            return disk_features[full_path]
        disk_features[full_path] = np.load(
            full_path,
            mmap_mode='r'
        )
        return disk_features[full_path]


def init_pytorch_worker(rank:int, world_size:int, manager_ip:str, manager_port:int) -> None:
    import cupy
    import rmm
    from pynvml.smi import nvidia_smi

    smi = nvidia_smi.getInstance()
    #pool_size = smi.DeviceQuery('memory.total')['gpu'][device_id]['fb_memory_usage']['total'] * 0.9 * 1e6
    pool_size=8e9

    rmm.reinitialize(
        devices=[rank],
        pool_allocator=True,
        initial_pool_size=pool_size,
    )

    #from rmm.allocators.torch import rmm_torch_allocator
    #torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

    from rmm.allocators.cupy import rmm_cupy_allocator
    cupy.cuda.set_allocator(rmm_cupy_allocator)

    cupy.cuda.Device(rank).use()
    torch.cuda.set_device(rank)

    # Pytorch training worker initialization
    dist_init_method = f"tcp://{manager_ip}:{manager_port}"

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=dist_init_method,
        world_size=world_size,
        rank=rank,
    )

def extend_tensor(t: torch.Tensor, l:int):
    return torch.concat([
        t,
        torch.zeros(
            l - len(t),
            dtype=t.dtype,
            device=t.device
        )
    ])

def train_epoch(model, loader, optimizer):
    total_loss = 0.0
    num_batches = 0

    time_forward = 0.0
    time_backward = 0.0
    time_loader = 0.0
    time_feature_additional = 0.0
    start_time = time.perf_counter()
    end_time_backward = start_time
    with td.algorithms.join.Join([model, optimizer]):
        for iter_i, data in enumerate(loader):
            loader_time_iter = time.perf_counter() - end_time_backward
            time_loader += loader_time_iter

            #data = data.to_homogeneous()
            num_sampled_nodes = data['paper']['num_sampled_nodes']
            num_sampled_edges = data['paper','cites','paper']['num_sampled_edges']

            num_layers = len(model.module.convs)
            num_sampled_nodes = extend_tensor(num_sampled_nodes, num_layers + 1)
            num_sampled_edges = extend_tensor(num_sampled_edges, num_layers)

            num_batches += 1
            if iter_i % 20 == 1:
                time_forward_iter = time_forward / num_batches
                time_backward_iter = time_backward / num_batches
                
                total_time_iter = (time.perf_counter() - start_time) / num_batches
                print(f"iteration {iter_i}")
                print(f"num sampled nodes: {num_sampled_nodes}")
                print(f"num sampled edges: {num_sampled_edges}")
                print(f"time forward: {time_forward_iter}")
                print(f"time backward: {time_backward_iter}")
                print(f"loader time: {loader_time_iter}")
                print(f"total time: {total_time_iter}")

            
            additional_feature_time_start = time.perf_counter()
            y_true = data['paper'].y.cuda() # train
            x = data['paper'].x.cuda().to(torch.float32)
            additional_feature_time_end = time.perf_counter()
            time_feature_additional += additional_feature_time_end - additional_feature_time_start

            start_time_forward = time.perf_counter()
            edge_index = data['paper','cites','paper'].edge_index if 'edge_index' in data['paper','cites','paper'] else data['paper','cites','paper'].adj_t
            
            y_pred = model(
                x,
                edge_index,
                num_sampled_nodes,
                num_sampled_edges,
            )
            
            end_time_forward = time.perf_counter()
            time_forward += end_time_forward - start_time_forward
            
            if y_pred.shape[0] > len(y_true):
                raise ValueError(f"illegal shape: {y_pred.shape}; {y_true.shape}")

            y_true = y_true[:y_pred.shape[0]]

            # temporary fix
            y_true += 1
            y_true = F.one_hot(
                y_true.to(torch.int64), num_classes=172
            ).to(torch.float32)            

            if y_true.shape != y_pred.shape:
                raise ValueError(
                    f'y_true shape was {y_true.shape} '
                    f'but y_pred shape was {y_pred.shape} '
                    f'in iteration {iter_i} '
                    f'on rank {y_pred.device.index}'
                )
            

            start_time_backward = time.perf_counter()
            loss = F.cross_entropy(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            end_time_backward = time.perf_counter()
            time_backward += end_time_backward - start_time_backward
            
    
    end_time = time.perf_counter()
    return total_loss, num_batches, ((end_time - start_time) / num_batches), (time_forward / num_batches), (time_backward / num_batches), (time_loader / num_batches), (time_feature_additional / num_batches)


def load_graph_native(bulk_samples_dir: str, features_device: Union[str,int] = "cpu"):
    with open(os.path.join(bulk_samples_dir, 'output_meta.json'), 'r') as f:
        output_meta = json.load(f)
    
    dataset_path = os.path.join(output_meta['dataset_dir'], output_meta['dataset'])
    with open(os.path.join(dataset_path, 'meta.json'), 'r') as f:
        input_meta = json.load(f)
    
    replication_factor = output_meta['replication_factor']
    num_nodes_dict = {node_type: t * replication_factor for node_type, t in input_meta['num_nodes'].items()}

    # Have to load graph data for native PyG
    parquet_path = os.path.join(
        output_meta['dataset_dir'],
        output_meta['dataset'],
        'parquet'
    )

    from torch_geometric.data import HeteroData
    hetero_data = HeteroData()

    num_input_features = 0
    num_output_features = 0
    print('loading feature data...')
    for node_type in os.listdir(os.path.join(dataset_path, 'npy')):
        feature_data = load_disk_features(output_meta, node_type, replication_factor=replication_factor)
        hetero_data[node_type].x = torch.as_tensor(feature_data, device=features_device)

        if feature_data.shape[1] > num_input_features:
            num_input_features = feature_data.shape[1]

        label_path = os.path.join(dataset_path, 'parquet', node_type, 'node_label.parquet')
        if os.path.exists(label_path):
            node_label = pandas.read_parquet(label_path)
            if replication_factor > 1:
                base_num_nodes = input_meta['num_nodes'][node_type]
                dfr = pandas.DataFrame({
                    'node': pandas.concat([node_label.node + (r * base_num_nodes) for r in range(1, replication_factor)]),
                    'label': pandas.concat([node_label.label for r in range(1, replication_factor)]),
                })
                node_label = pandas.concat([node_label, dfr]).reset_index(drop=True)

            node_label_tensor = torch.full((num_nodes_dict[node_type],), -1, dtype=torch.float32, device='cpu')
            node_label_tensor[torch.as_tensor(node_label.node.values, device='cpu')] = \
                torch.as_tensor(node_label.label.values, device='cpu')
            
            del node_label
            gc.collect()

            hetero_data[node_type]['train'] = (node_label_tensor > -1).contiguous()
            hetero_data[node_type]['y'] = node_label_tensor.contiguous()
            hetero_data[node_type]['num_nodes'] = num_nodes_dict[node_type]

            num_classes = int(node_label_tensor.max()) + 1
            if num_classes > num_output_features:
                num_output_features = num_classes

        print('done loading feature data')

    for edge_type in input_meta['num_edges'].keys():
        print(f'Loading edge index for edge type {edge_type}')

        print('reading parquet file...')
        can_edge_type = tuple(edge_type.split('__'))
        ei = pandas.read_parquet(os.path.join(os.path.join(parquet_path, edge_type), 'edge_index.parquet'))
        ei = {
            'src': torch.as_tensor(ei.src.values, device='cpu'),
            'dst': torch.as_tensor(ei.dst.values, device='cpu'),
        }

        print('sorting edge index...')
        ei['dst'], ix = torch.sort(ei['dst'])
        ei['src'] = ei['src'][ix]
        del ix
        gc.collect()

        print('processing replications...')
        if replication_factor > 1:
            orig_src = ei['src'].clone().detach()
            orig_dst = ei['dst'].clone().detach()
            for r in range(1, replication_factor):
                ei['src'] = torch.concat([
                    ei['src'],
                    orig_src + int(r * input_meta['num_nodes'][can_edge_type[0]]),
                ])

                ei['dst'] = torch.concat([
                    ei['dst'],
                    orig_dst + int(r * input_meta['num_nodes'][can_edge_type[2]]),
                ])

            del orig_src
            del orig_dst

            ei['src'] = ei['src'].contiguous()
            ei['dst'] = ei['dst'].contiguous()
        gc.collect()

        print(f"# edges: {len(ei['src'])}")

        print('converting to csc...')
        from torch_geometric.utils.sparse import index2ptr
        ei['dst'] = index2ptr(ei['dst'], num_nodes_dict[can_edge_type[2]])

        print('updating data structure...')
        hetero_data.put_edge_index(
            layout='csc',
            edge_index=list(ei.values()),
            edge_type=can_edge_type,
            size=(num_nodes_dict[can_edge_type[0]], num_nodes_dict[can_edge_type[2]]),
            is_sorted=True
        )
        gc.collect()

    print('done loading graph structure and features')    
    return hetero_data, num_input_features, num_output_features
    

def train_native(hetero_data, model_shape, bulk_samples_dir: str, device:int, features_device:Union[str, int] = "cpu", world_size=1, num_epochs=1) -> None:
    from models_native import GraphSAGE
    from torch_geometric.loader import NeighborLoader

    num_input_features, num_output_features = model_shape

    with open(os.path.join(bulk_samples_dir, 'output_meta.json'), 'r') as f:
        output_meta = json.load(f)

    with torch.cuda.device(device):
        td.barrier()

        #hetero_data['paper']['train'] = hetero_data['paper']['train'].cuda()
        #hetero_data['paper']['y'] = hetero_data['paper']['y'].cuda()

        model = GraphSAGE(
                in_channels=num_input_features,
                hidden_channels=64,
                out_channels=num_output_features,
                num_layers=len(output_meta['fanout'])
        ).to(torch.float32).to(device)
        
        td.barrier()
        model = ddp(model, device_ids=[device])
        print('done creating model')

        #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer = ZeroRedundancyOptimizer(model.parameters(), torch.optim.Adam, lr=0.01)

        for epoch in range(num_epochs):
            start_time_train = time.perf_counter_ns()
            model.train()
            
            print('creating loader...')
            with td.algorithms.join.Join([model]):
                input_nodes = torch.arange(hetero_data['paper']['num_nodes'])
                input_nodes = input_nodes[hetero_data['paper']['train']]
                input_nodes = torch.as_tensor(
                    np.array_split(
                        np.asarray(input_nodes),
                        world_size
                    )[device]
                )
                print('input nodes shape: ', input_nodes.shape)
                loader = NeighborLoader(
                    hetero_data,
                    input_nodes=('paper', input_nodes.cpu()),
                    batch_size=output_meta['batch_size'],
                    num_neighbors={('paper','cites','paper'):output_meta['fanout']},
                    replace=False,
                    is_sorted=True,
                    disjoint=False,
                    num_workers=4,
                    persistent_workers=True,
                )

            td.barrier()
            print('done creating loader')
            
            # loader was patched to record the feature extraction time

            total_loss, num_batches, mean_total_time, mean_forward_time, mean_backward_time, mean_loader_time, mean_additional_feature_time = \
                train_epoch(model, loader, optimizer)

            end_time_train = time.perf_counter_ns()
            print(
                f"epoch {epoch} time: "
                f"{(end_time_train - start_time_train) / 1e9:3.4f} s"
                f"\n trained {num_batches} batches"
            )
            print(f"loss after epoch {epoch}: {total_loss / num_batches}")
    
    return mean_total_time, mean_forward_time, mean_backward_time, (loader._feature_time / num_batches) + mean_additional_feature_time, (mean_loader_time - (loader._feature_time / num_batches))

def train(bulk_samples_dir: str, output_dir:str, native_times:List[float], device: int, features_device: Union[str, int] = "cpu", world_size=1, num_epochs=1) -> None:
    """
    Parameters
    ----------
    device: int
        The CUDA device where the model, graph data, and node labels will be stored.
    features_device: Union[str, int]
        The device (CUDA device or CPU) where features will be stored.
    """

    import cudf
    import cugraph
    from cugraph_pyg.data import CuGraphStore
    from cugraph_pyg.loader import BulkSampleLoader

    with torch.cuda.device(device):

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
        for node_type in input_meta['num_nodes'].keys():
            feature_data = load_disk_features(output_meta, node_type, replication_factor=replication_factor)
            print(f'features shape: {feature_data.shape}')
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
                if replication_factor > 1:
                    base_num_nodes = input_meta['num_nodes'][node_type]
                    print('base num nodes:', base_num_nodes)
                    dfr = cudf.DataFrame({
                        'node': cudf.concat([node_label.node + (r * base_num_nodes) for r in range(1, replication_factor)]),
                        'label': cudf.concat([node_label.label for r in range(1, replication_factor)]),
                    })
                    node_label = cudf.concat([node_label, dfr]).reset_index(drop=True)

                node_label_tensor = torch.full((N[node_type],), -1, dtype=torch.float32, device='cuda')
                node_label_tensor[torch.as_tensor(node_label.node.values, device='cuda')] = \
                    torch.as_tensor(node_label.label.values, device='cuda')
                
                del node_label
                gc.collect()

                fs.add_data((node_label_tensor > -1).contiguous(), node_type, 'train')
                fs.add_data(node_label_tensor.contiguous(), node_type, 'y')
                num_classes = int(node_label_tensor.max()) + 1
                if num_classes > num_output_features:
                    num_output_features = num_classes
        print('done loading data')
        td.barrier()

        print(f"num input features: {num_input_features}; num output features: {num_output_features}; fanout: {output_meta['fanout']}")
        
        num_hidden_channels = 64
        num_layers = len(output_meta['fanout'])
        model = CuGraphSAGE(
                in_channels=num_input_features,
                hidden_channels=num_hidden_channels,
                out_channels=num_output_features,
                num_layers=num_layers
        ).to(torch.float32).to(device)
        
        model = ddp(model, device_ids=[device])
        
        print('done creating model')
        td.barrier()
        
        cugraph_store = CuGraphStore(fs, G, N)
        print('done creating store')
        td.barrier()

        #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer = ZeroRedundancyOptimizer(model.parameters(), torch.optim.Adam, lr=0.01)
        td.barrier()

        for epoch in range(num_epochs):
            start_time_train = time.perf_counter_ns()
            model.train()

            print('creating loader...')
            samples_dir = os.path.join(bulk_samples_dir, 'samples')
            input_files = np.array(os.listdir(samples_dir))
            input_files = np.array_split(
                input_files, world_size
            )[device].tolist()

            cugraph_loader = BulkSampleLoader(
                cugraph_store,
                cugraph_store,
                input_nodes=None,
                input_files=input_files,
                directory=samples_dir,
            )
            print('done creating loader')
            td.barrier()

            total_loss, num_batches, mean_total_time, mean_time_fw, mean_time_bw, mean_time_loader, mean_additional_feature_time = train_epoch(model, cugraph_loader, optimizer)

            end_time_train = time.perf_counter_ns()
            train_time = (end_time_train - start_time_train) / 1e9
            print(
                f"epoch {epoch} time: "
                f"{train_time:3.4f} s"
                f"\n trained {num_batches} batches"
            )
            print(f"loss after epoch {epoch}: {total_loss / num_batches}")

        train_time = mean_total_time * num_batches
        output_result_filename = f'results[{device}].csv'
        results_native = {
            'Dataset': f"{output_meta['dataset']} x {replication_factor}",
            'Framework': 'PyG',
            'Setup Details': f"GraphSAGE, {num_layers} layers",
            'Batch Size': output_meta['batch_size'],
            'Fanout': str(output_meta['fanout']),
            'Machine Details': socket.gethostname(),
            'Sampling per epoch': native_times[4] * num_batches,
            'MFG Creation': 0.0,
            'Feature Loading': native_times[3] * num_batches,
            'Model FWD': native_times[1] * num_batches,
            'Model BWD': native_times[2] * num_batches,
            'Time Per Epoch': native_times[0] * num_batches,
            'Time Per Batch': native_times[0],
            'Speedup': 1,
        }
        results_cugraph = {
            'Dataset': f"{output_meta['dataset']} x {replication_factor}",
            'Framework': 'cuGraph-PyG',
            'Setup Details': f"GraphSAGE, {num_layers} layers",
            'Batch Size': output_meta['batch_size'],
            'Fanout': str(output_meta['fanout']),
            'Machine Details': socket.gethostname(),
            'Sampling per epoch': output_meta['execution_time'],
            'MFG Creation': cugraph_loader._total_convert_time + cugraph_loader._total_read_time,
            'Feature Loading': cugraph_loader._total_feature_time + (mean_additional_feature_time * num_batches),
            'Model FWD': mean_time_fw * num_batches,
            'Model BWD': mean_time_bw * num_batches,
            'Time Per Epoch': train_time + output_meta['execution_time'],
            'Time Per Batch': (train_time + output_meta['execution_time']) / num_batches,
            'Speedup': (native_times[0] * num_batches) / (train_time + output_meta['execution_time']),
        }
        results = {
            'Machine': socket.gethostname(),
            'Comms': output_meta['comms'] if 'comms' in output_meta else 'tcp',
            'Dataset': output_meta['dataset'],
            'Replication Factor': replication_factor,
            'Model': 'GraphSAGE',
            '# Layers': num_layers,
            '# Input Channels': num_input_features,
            '# Output Channels': num_output_features,
            '# Hidden Channels': num_hidden_channels,
            '# Vertices': output_meta['total_num_nodes'],
            '# Edges': output_meta['total_num_edges'],
            '# Vertex Types': len(N.keys()),
            '# Edge Types': len(G.keys()),
            'Sampling # GPUs': output_meta['num_sampling_gpus'],
            'Seeds Per Call': output_meta['seeds_per_call'],
            'Batch Size': output_meta['batch_size'],
            '# Train Batches': num_batches,
            'Batches Per Partition': output_meta['batches_per_partition'],
            'Fanout': str(output_meta['fanout']),
            'Training # GPUs': 1,
            'Feature Storage': 'cpu' if features_device == 'cpu' else 'gpu',
            'Memory Type': 'Device', # could be managed if configured

            'Total Time': train_time + output_meta['execution_time'],
            'Native Equivalent Time': native_times[0] * num_batches,
            'Total Speedup': (native_times[0] * num_batches) / (train_time + output_meta['execution_time']),

            'Bulk Sampling Time': output_meta['execution_time'],
            'Bulk Sampling Time Per Batch': output_meta['execution_time'] / num_batches,

            'Parquet Read Time': cugraph_loader._total_read_time,
            'Parquet Read Time Per Batch': cugraph_loader._total_read_time / num_batches,

            'Minibatch Conversion Time': cugraph_loader._total_convert_time,
            'Minibatch Conversion Time Per Batch': cugraph_loader._total_convert_time / num_batches,

            'Feature Fetch Time': cugraph_loader._total_feature_time,
            'Feature Fetch Time Per Batch': cugraph_loader._total_feature_time / num_batches,

            'Foward Time': mean_time_fw * num_batches,
            'Native Forward Time': native_times[1] * num_batches,

            'Forward Time Per Batch': mean_time_fw,
            'Native Forward Time Per Batch': native_times[1],

            'Backward Time': mean_time_bw * num_batches,
            'Native Backward Time': native_times[2] * num_batches,

            'Backward Time Per Batch': mean_time_bw,
            'Native Backward Time Per Batch': native_times[2],
        }
        df = pandas.DataFrame(results, index=[0])
        df.to_csv(os.path.join(output_dir, output_result_filename),header=True, sep=',', index=False, mode='a')

        df_n = pandas.DataFrame(results_native, index=[0])
        df_c = pandas.DataFrame(results_cugraph, index=[1])
        pandas.concat([df_n, df_c]).to_csv(os.path.join(output_dir, output_result_filename),header=True, sep=',', index=False, mode='a')
        
        print('convert:', cugraph_loader._total_convert_time)
        print('read:', cugraph_loader._total_read_time)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features_device",
        type=str,
        default="cpu",
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

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to store results",
        required=True
    )

    parser.add_argument(
        "--native_times",
        type=str,
        help="Input the native runtimes (total, fw, bw) to avoid doing a native run",
        required=False,
        default="-1.0,-1.0,-1.0,-1.0,-1.0"
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        help="The number of GPUs to allocate to PyTorch",
        required=False,
        default=1
    )

    return parser.parse_args()


def main(rank, args, hetero_data, num_input_features, num_output_features):
    device = rank
    features_device = 'cpu' if args.features_device == 'cpu' else device
    init_pytorch_worker(device, args.num_gpus, "127.0.0.1", 12345)
    enable_spilling()
    print(f'rank {rank} initialized')
    td.barrier()

    native_mean_time, native_mean_fw_time, native_mean_bw_time, native_mean_feature_time, native_mean_sample_time = [float(x) for x in args.native_times.split(',')]
    if native_mean_time < 0:
        native_mean_time, native_mean_fw_time, native_mean_bw_time, native_mean_feature_time, native_mean_sample_time = \
            train_native(
                hetero_data,
                (num_input_features, num_output_features),
                args.sample_dir,
                device=device,
                features_device=features_device,
                world_size=args.num_gpus,
                num_epochs=args.num_epochs
            )
        
    train(
        args.sample_dir,
        args.output_dir,
        (native_mean_time, native_mean_fw_time, native_mean_bw_time, native_mean_feature_time, native_mean_sample_time),
        device=device,
        features_device=features_device,
        world_size=args.num_gpus,
        num_epochs=args.num_epochs
    )


if __name__ == "__main__":
    args = parse_args()

    hetero_data, num_input_features, num_output_features = None,None,None
    if False:
        hetero_data, num_input_features, num_output_features = load_graph_native(
            args.sample_dir,
            features_device="cpu"
        )

    tmp.spawn(
        main,
        args=[args, hetero_data, num_input_features, num_output_features],
        nprocs=args.num_gpus,
        join=True,
    )

