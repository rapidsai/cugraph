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

# This is a multi-GPU benchmark that assumes the data has already been
# processed using the BulkSampler.  This workflow WILL ONLY WORK when
# reading already-processed sampling results from disk.

import re
import json
import argparse
import os
from math import ceil

import numpy as np
import pandas as pd

import torch
import torch.distributed as td
import torch.multiprocessing as tmp
from torch.nn.parallel import DistributedDataParallel

from cugraph.gnn import FeatureStore
from cugraph_pyg.data import CuGraphStore
from cugraph_pyg.loader import BulkSampleLoader
from distributed import Client, Event as Dask_Event

def load_features(node_type_name, features_path, num_nodes):
    F = FeatureStore(backend='torch')
    node_features = np.load(
        os.path.join(features_path, 'node_feat.npy'),
        mmap_mode='r'
    )
    F.add_data(node_features, node_type_name, 'x')

    node_labels_train = pd.read_parquet(
        os.path.join(features_path, 'train_labels')
    )
    node_labels_train['train'] = True
    node_labels_test = pd.read_parquet(
        os.path.join(features_path, 'test_labels')
    )
    node_labels_test['test'] = True
    node_labels_val = pd.read_parquet(
        os.path.join(features_path, 'val_labels')
    )
    node_labels_val['val'] = True

    all_labels = pd.concat([
        node_labels_train,
        node_labels_test,
        node_labels_val
    ], ignore_index=True)
    all_labels = all_labels.set_index('node')

    bdf = pd.DataFrame(index=np.arange(num_nodes))
    # This will upcast the label to float64, but that is ok in this case.
    all_labels = bdf.join(all_labels, how='left').fillna({
        'train':False,
        'test':False,
        'val':False,
        'label': -1,
    })

    # At this point, all_labels holds the train/test/val mask and labels
    # Unlabeled nodes get the default label (class id) of -1.
    # Users should never be accessing labels of unlabeled nodes - they
    # should filter through the train/test/val mask first.
    for col in all_labels:
        F.add_data(all_labels[col].to_numpy(), node_type_name, col)
    
    return F

def get_batches_per_partition(dir):
    sample_fname = os.listdir(dir)[0]
    m = re.match(r'batch\=([0-9]+)\-([0-9]+)\.parquet', sample_fname)
    if m is None:
        raise ValueError(f'Unexpected partition schema in {dir}')
    
    return int(m[1]) - int(m[0]) + 1

def enable_cudf_spilling():
    import cudf
    cudf.set_option('spill', True)

def init_pytorch_worker(rank, devices, manager_ip, manager_port):
    import cupy
    import rmm

    # The pytorch device is set through a context manager

    # Set cupy to the correct device
    cupy.cuda.Device(devices[rank]).use()

    # Pytorch training worker initialization
    dist_init_method = f"tcp://{manager_ip}:{manager_port}"

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=dist_init_method,
        world_size=len(devices),
        rank=rank,
    )

    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=10e9,
        maximum_pool_size=15e9,
        devices=devices[rank]
    )

    cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)
    # FIXME eventually will probably set torch's allocator here

    enable_cudf_spilling()

def terminate_pytorch_worker():
    pass

def start_cugraph_dask_client(dask_scheduler_address):
    from cugraph.dask.comms import comms as Comms

    client = Client(dask_scheduler_address)
    Comms.initialize(p2p=True)
    return client

def run(rank, devices, dask_scheduler_address, manager_ip, manager_port, features_path, sampled_data_path):
    device = devices[rank]
    torch_device = torch.device(f"cuda:{device}")

    with torch.cuda.device(torch_device):
        init_pytorch_worker(rank, devices, manager_ip, manager_port)
        client = start_cugraph_dask_client(dask_scheduler_address)
        event = Dask_Event("cugraph_store_creation_event")

        if rank == 0:
            # create the feature store, graph store
            meta_json_path = os.path.join(sampled_data_path, 'meta.json')
            with open(meta_json_path, 'r') as f:
                meta = json.load(f)
                
            num_papers = (
                meta['node_id_range_paper'][1]
                - meta['node_id_range_paper'][0]
                + 1
            )

            num_epochs = meta['number_of_epochs']
            num_train_batches = meta['number_of_train_batches']
            num_test_batches = meta['number_of_test_batches']
            num_val_batches = meta['number_of_val_batches']

            # For now there is only the "paper" node type
            F = load_features('paper', features_path, num_papers)
            G = {
                ('paper', 'cites', 'paper'): meta['number_of_edges_paper__cites__paper']
            }
            N = {
                'paper': num_papers
            }

            cugraph_store = CuGraphStore(F, G, N)
            client.publish_dataset(cugraph_store=cugraph_store)
            client.publish_dataset(num_train_batches=num_train_batches)
            client.publish_dataset(num_test_batches=num_test_batches)
            client.publish_dataset(num_val_batches=num_val_batches)
            client.publish_dataset(num_epochs=num_epochs)
            event.set()
            # No need to publish anything else, train/test/val masks are in
            # features store
        else:
            if event.wait(timeout=1000):
                cugraph_store = client.get_dataset('cugraph_store')
                num_train_batches = client.get_dataset('num_train_batches')
                num_test_batches = client.get_dataset('num_test_batches')
                num_val_batches = client.get_dataset('num_val_batches')
                num_epochs = client.get_dataset('num_epochs')
            else:
                raise RuntimeError(f'Rank {rank} was unable to load cugraph store!')
    
        td.barrier()

        for epoch in range(num_epochs):
            # Create the loaders
            epoch_path = os.path.join(
                os.path.join(sampled_data_path, 'train_{epoch}'),
                'rank=0'
            )
            
            train_batches_per_partition = get_batches_per_partition(epoch_path)
            num_partitions_per_rank = num_train_batches // train_batches_per_partition
            remainder = num_train_batches % train_batches_per_partition
            train_batches_per_rank = num_partitions_per_rank * train_batches_per_partition
            starting_batch_id = train_batches_per_rank * rank
            if rank == len(devices) - 1:
                train_batches_per_rank += remainder

            cugraph_bulk_loader = BulkSampleLoader(
                cugraph_store,
                cugraph_store,
                train_batches_per_rank,
                starting_batch_id=starting_batch_id,
                batches_per_partition=train_batches_per_partition,
                directory=epoch_path, # SET THIS CORRECTLY
            )
            for hetero_data in cugraph_bulk_loader:
                # train
                pass

        td.barrier()

    # cleanup dask cluster
    if rank == 0:
        client.unpublish_dataset("cugraph_store")
        client.unpublish_dataset('num_train_batches')
        client.unpublish_dataset('num_test_batches')
        client.unpublish_dataset('num_val_batches')
        client.unpublish_dataset('num_epochs')
        event.clear()
        

def setup_cluster(dask_worker_devices):
    dask_worker_devices_str = ",".join([str(i) for i in dask_worker_devices])
    from dask_cuda import LocalCUDACluster

    cluster = LocalCUDACluster(
        protocol="tcp",
        CUDA_VISIBLE_DEVICES=dask_worker_devices_str,
        rmm_pool_size="25GB",
    )

    client = Client(cluster)
    client.wait_for_workers(n_workers=len(dask_worker_devices))
    client.run(enable_cudf_spilling)
    print("Dask Cluster Setup Complete")
    del client
    return cluster.scheduler_address

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torch_devices",
        type=str,
        default="0,1,2",
        help='GPUs to allocate to pytorch',
        required=False
    )

    parser.add_argument(
        "--dask_devices",
        type=str,
        default="3",
        help='GPUs to allocate to dask',
        required=False
    )

    parser.add_argument(
        '--torch_manager_ip',
        type=str,
        default="127.0.0.1",
        help='The torch distributed manager ip address',
        required=False,
    )

    parser.add_argument(
        '--torch_manager_port',
        type=str,
        default="12346",
        help='The torch distributed manager port',
        required=False,
    )

    parser.add_argument(
        "--features_path",
        type=str,
        help='The path to the node features',
        required=True
    )

    parser.add_argument(
        "--sampled_data_path",
        type=str,
        help='The path to the sampled data',
        required=True
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # devices, dask_scheduler_address, manager_ip, manager_port, features_path, sampled_data_path
    torch_devices = [int(d) for d in args.torch_devices.split(',')]
    dask_devices = [int(d) for d in args.dask_devices.split(',')]

    dask_scheduler_address = setup_cluster(dask_devices)

    run_args = (
        torch_devices,
        dask_scheduler_address,
        args.torch_manager_ip,
        args.torch_manager_port,
        args.features_path,
        args.sampled_data_path,   
    )

    tmp.spawn(
        run,
        args=run_args,
        nprocs=len(torch_devices),
        join=True
    )

    pass

if __name__ == '__main__':
    main()