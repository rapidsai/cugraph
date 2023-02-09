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
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as td
import torch.multiprocessing as tmp
from torch.nn.parallel import DistributedDataParallel as ddp
import torchmetrics.functional as MF

from torch_geometric.nn import SAGEConv

from cugraph.gnn import FeatureStore
from cugraph_pyg.data import CuGraphStore
from cugraph_pyg.loader import BulkSampleLoader
from cugraph.dask.common.mg_utils import teardown_local_dask_cluster
from distributed import Client, Event as Dask_Event

def load_features(node_type_name, features_path, num_nodes):
    fs = FeatureStore(backend='torch')
    node_features = np.load(
        os.path.join(features_path, 'node_feat.npy'),
        mmap_mode='r'
    )
    fs.add_data(node_features, node_type_name, 'x')

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
    }).rename(columns={'label': 'y'})

    # At this point, all_labels holds the train/test/val mask and labels
    # Unlabeled nodes get the default label (class id) of -1.
    # Users should never be accessing labels of unlabeled nodes - they
    # should filter through the train/test/val mask first.
    for col in all_labels:
        fs.add_data(all_labels[col].to_numpy(), node_type_name, col)
    
    return fs

def get_batches_per_partition(dir):
    dir = os.path.join(dir, 'rank=0')
    sample_fname = os.listdir(dir)[0]
    m = re.match(r'batch\=([0-9]+)\-([0-9]+)\.parquet', sample_fname)
    if m is None:
        raise ValueError(f'Unexpected partition schema in {dir}')
    
    return int(m[2]) - int(m[1]) + 1

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


class SAGE(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = SAGEConv((-1, -1), hidden_channels)
            self.convs.append(conv)

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge):
        for conv in self.convs:
            x = conv(x, edge)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=0.5)
        return self.lin(x)

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
            fs = load_features('paper', features_path, num_papers)
            G = {
                ('paper', 'cites', 'paper'): meta['number_of_edges_paper__cites__paper']
            }
            N = {
                'paper': num_papers
            }

            import pickle
            cugraph_store = CuGraphStore(fs, G, N)
            client.publish_dataset(cugraph_store=pickle.dumps(cugraph_store))
            client.publish_dataset(num_train_batches=num_train_batches)
            client.publish_dataset(num_test_batches=num_test_batches)
            client.publish_dataset(num_val_batches=num_val_batches)
            client.publish_dataset(num_epochs=num_epochs)
            event.set()
            # No need to publish anything else, train/test/val masks are in
            # features store
        else:
            if event.wait(timeout=1000):
                import pickle
                cugraph_store = pickle.loads(client.get_dataset('cugraph_store'))
                num_train_batches = client.get_dataset('num_train_batches')
                num_test_batches = client.get_dataset('num_test_batches')
                num_val_batches = client.get_dataset('num_val_batches')
                num_epochs = client.get_dataset('num_epochs')
            else:
                raise RuntimeError(f'Rank {rank} was unable to load cugraph store!')
    
        td.barrier()

        model = SAGE(hidden_channels=64, out_channels=150,
                  num_layers=2).to(torch.float64).to(torch_device)
        
        with torch.no_grad():  # Initialize lazy modules.
            dummy_ei = [
                torch.randint(0,200,(1000,), dtype=torch.long),
                torch.randint(0,200,(1000,), dtype=torch.long)
            ]
            dummy_n = torch.concat(dummy_ei).unique().to(torch_device)
            dummy_x = cugraph_store.get_tensor('paper','x', dummy_n)
            model(dummy_x, torch.stack(dummy_ei).to(torch_device))

        model = ddp(model, device_ids=[rank])

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        td.barrier()

        for epoch in range(num_epochs):
            model.train()
            # Create the loaders
            epoch_path = os.path.join(sampled_data_path, f'train_{epoch}')
            
            train_batches_per_partition = get_batches_per_partition(epoch_path)
            num_partitions_per_epoch = int(ceil(num_train_batches / train_batches_per_partition))
            
            num_workers = len(devices)
            if num_workers > num_partitions_per_epoch:
                raise RuntimeError(f"Too many workers (num workers {num_workers} > num partitions ({num_partitions_per_epoch})")
            
            num_partitions_per_rank = num_partitions_per_epoch // num_workers

            train_batches_per_rank = num_partitions_per_rank * train_batches_per_partition
            remainder = num_train_batches - (train_batches_per_rank * num_workers)
            starting_batch_id = train_batches_per_rank * rank
            if rank == num_workers - 1:
                train_batches_per_rank += remainder

            print(f'{rank} partitions per rank: ', num_partitions_per_rank, flush=True)
            print(f'{rank} train batches per rank: ', train_batches_per_rank, flush=True)
            print(f'{rank} starting batch id: ', starting_batch_id, flush=True)
            cugraph_bulk_loader = BulkSampleLoader(
                cugraph_store,
                cugraph_store,
                train_batches_per_rank,
                starting_batch_id=starting_batch_id,
                batches_per_partition=train_batches_per_partition,
                directory=epoch_path,
            )

            total_loss = 0
            num_batches = 0
            # This context manager will handle different # batches per rank
            # barrier() cannot do this since the number of ops per rank is
            # different.  It essentially acts like barrier would if the
            # number of ops per rank was the same.
            with torch.distributed.algorithms.join.Join([model]):
                for hetero_data in cugraph_bulk_loader:
                    print(hetero_data)
                    num_batches += 1
                    # train
                    train_mask = hetero_data.train_dict['paper'].to(torch_device)
                    y_true = hetero_data.y_dict['paper']
                    y_pred = model(
                        hetero_data.x_dict['paper'].to(torch_device),
                        hetero_data.edge_index_dict[('paper','cites','paper')].to(torch_device)
                    )

                    y_true = F.one_hot(
                        y_true[train_mask].to(torch.int64),
                        num_classes=150
                    ).to(torch.float64)
                    y_pred = y_pred[train_mask]
                    
                    print(y_pred.shape)
                    print(y_true.shape)
                    loss = F.cross_entropy(
                        y_pred,
                        y_true
                    )
                    optimizer.zero_grad()
                    print('loss:', loss)
                    loss.backward()
                    print('backward')
                    optimizer.step()
                    print('step')
                    total_loss += loss.item()
            
            print(f'{rank} done training for epoch {epoch}')

            # test
            print('TEST')
            model.eval()
            if rank == 0:                
                eval_path = os.path.join(sampled_data_path, 'test')
                test_batches_per_partition = get_batches_per_partition(eval_path)
                cugraph_bulk_eval_loader = BulkSampleLoader(
                    cugraph_store,
                    cugraph_store,
                    num_test_batches,
                    starting_batch_id=0,
                    batches_per_partition=test_batches_per_partition,
                    directory=eval_path
                )

                y_pred = []
                y_true = []
                for eval_hetero_data in cugraph_bulk_eval_loader:
                    with torch.no_grad():
                        test_mask = eval_hetero_data.test_dict['paper']
                        y_true_i = eval_hetero_data.y_dict['paper']
                        y_true.append(y_true_i[test_mask])
                        
                        y_pred_i = model(
                            hetero_data.x_dict['paper'],
                            hetero_data.edge_index_dict[('paper','cites','paper')]
                        )
                        y_pred.append(y_pred_i[test_mask])
                
                num_classes = 150
                acc = MF.accuracy(
                    torch.cat(y_pred.to(torch.int64)),
                    torch.pred(F.one_hot(y_true, num_classes=150).to(torch.int64)),
                    task='multiclass',
                    num_classes=num_classes
                )

                print(f'epoch: {epoch:05d} | loss: {(total_loss / num_batches):7.2f} | acc: {acc:.4f}')

        td.barrier()

    # cleanup dask cluster
    td.barrier()
    if rank == 0:
        print("DONE", flush=True)
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


if __name__ == '__main__':
    main()