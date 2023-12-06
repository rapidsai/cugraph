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
from trainers_cugraph import CuGraphTrainer
from trainers_native import NativeTrainer

from datasets import OGBNPapers100MDataset

from cugraph.testing.mg_utils import enable_spilling

def init_pytorch_worker(rank: int, use_rmm_torch_allocator: bool=False) -> None:
    import cupy
    import rmm
    from pynvml.smi import nvidia_smi

    smi = nvidia_smi.getInstance()
    pool_size=8e9 # FIXME calculate this

    rmm.reinitialize(
        devices=[rank],
        pool_allocator=True,
        initial_pool_size=pool_size,
    )

    if use_rmm_torch_allocator:
        from rmm.allocators.torch import rmm_torch_allocator
        torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

    from rmm.allocators.cupy import rmm_cupy_allocator
    cupy.cuda.set_allocator(rmm_cupy_allocator)

    cupy.cuda.Device(rank).use()
    torch.cuda.set_device(rank)

    # Pytorch training worker initialization
    torch.distributed.init_process_group(backend="nccl")

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
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
        required=False,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size (required for Native run)",
        required=False,
    )

    parser.add_argument(
        "--fanout",
        type=str,
        default="10_10_10",
        help="Fanout (required for Native run)",
        required=False,
    )

    parser.add_argument(
        "--sample_dir",
        type=str,
        help="Directory with stored bulk samples (required for cuGraph run)",
        required=False,
    )

    parser.add_argument(
        "--output_file",
        type=str,
        help="File to store results",
        required=True,
    )

    parser.add_argument(
        "--framework",
        type=str,
        help="The framework to test (cuGraph or Native)",
        required=True,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="GraphSAGE",
        help="The model to use (currently only GraphSAGE supported)",
        required=False,
    )

    parser.add_argument(
        '--replication_factor',
        type=int,
        default=1,
        help="The replication factor for the dataset",
        required=False,
    )

    parser.add_argument(
        '--dataset_dir',
        type=str,
        help="The directory where datasets are stored",
        required=True,
    )

    parser.add_argument(
        '--train_split',
        type=float,
        help="The percentage of the labeled data to use for training.  The remainder is used for testing/validation.",
        default=0.8,
        required=False,
    )

    parser.add_argument(
        '--val_split',
        type=float,
        help="The percentage of the testing/validation data to allocate for validation.",
        default=0.5,
        required=False,
    )

    return parser.parse_args()


def main(args):
    rank = int(os.environ['LOCAL_RANK'])

    init_pytorch_worker(rank, use_rmm_torch_allocator=(args.framework == "cuGraph"))
    enable_spilling()
    print(f'worker initialized')
    td.barrier()

    world_size = int(os.environ['SLURM_JOB_NUM_NODES']) * int(os.environ['SLURM_GPUS_PER_NODE'])

    dataset = OGBNPapers100MDataset(
        replication_factor=args.replication_factor,
        dataset_dir=args.dataset_dir,
        train_split=args.train_split,
        val_split=args.val_split,
    )

    if args.framework == "Native":
        trainer = NativeTrainer(
            model=args.model,
            dataset=dataset,
            device=rank,
            rank=rank,
            world_size=world_size,
            num_epochs=args.num_epochs,
            shuffle=True,
            replace=False,
            fanout=[int(f) for f in args.fanout.split('_')],
            batch_size=args.batch_size,
        )
    else:
        raise ValueError("unsuported framework")

    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    main(args)

