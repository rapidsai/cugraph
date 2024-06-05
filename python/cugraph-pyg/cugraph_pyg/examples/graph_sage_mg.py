# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

# For this script, dask must be started first in a separate process.
# To do this, the `start_dask.sh` script has been provided.  This scripts starts
# a dask scheduler and dask workers.  To select the GPUs and amount of memory
# allocated to dask per GPU, the `CUDA_VISIBLE_DEVICES` and `WORKER_RMM_POOL_SIZE`
# arguments in that script can be modified.
# To connect to dask, the scheduler JSON file must be provided.  This can be done
# using the `--dask_scheduler_file` argument in the mg python script being run.

from ogb.nodeproppred import NodePropPredDataset

import time
import argparse
import gc
import warnings

import torch
import numpy as np

from cugraph_pyg.nn import SAGEConv as CuGraphSAGEConv

import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as td
import torch.multiprocessing as tmp
from torch.nn.parallel import DistributedDataParallel as ddp

from typing import List


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


def enable_cudf_spilling():
    import cudf

    cudf.set_option("spill", True)


def init_pytorch_worker(rank, devices, manager_ip, manager_port) -> None:
    import cupy
    import rmm

    device_id = devices[rank]

    rmm.reinitialize(
        devices=[device_id],
        pool_allocator=False,
    )

    # torch.cuda.change_current_allocator(rmm.rmm_torch_allocator)
    # cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

    cupy.cuda.Device(device_id).use()
    torch.cuda.set_device(device_id)

    # Pytorch training worker initialization
    dist_init_method = f"tcp://{manager_ip}:{manager_port}"

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=dist_init_method,
        world_size=len(devices),
        rank=rank,
    )

    # enable_cudf_spilling()


def start_cugraph_dask_client(rank, dask_scheduler_file):
    print(
        "Connecting to dask... "
        "(warning: this may take a while depending on your configuration)"
    )
    start_time_connect_dask = time.perf_counter_ns()
    from distributed import Client
    from cugraph.dask.comms import comms as Comms

    client = Client(scheduler_file=dask_scheduler_file)
    Comms.initialize(p2p=True)

    end_time_connect_dask = time.perf_counter_ns()
    print(
        f"Successfully connected to dask on rank {rank}, took "
        f"{(end_time_connect_dask - start_time_connect_dask) / 1e9:3.4f} s"
    )
    return client


def stop_cugraph_dask_client():
    from cugraph.dask.comms import comms as Comms

    Comms.destroy()

    from dask.distributed import get_client

    get_client().close()


def train(
    rank,
    torch_devices: List[int],
    manager_ip: str,
    manager_port: int,
    dask_scheduler_file: str,
    num_epochs: int,
    features_on_gpu=True,
) -> None:
    """
    Parameters
    ----------
    device: int
        The CUDA device where the model, graph data, and node labels will be stored.
    features_on_gpu: bool
        Whether to store a replica of features on each worker's GPU.  If False,
        all features will be stored on the CPU.
    """

    start_time_preprocess = time.perf_counter_ns()

    world_size = len(torch_devices)
    device_id = torch_devices[rank]
    features_device = device_id if features_on_gpu else "cpu"
    init_pytorch_worker(rank, torch_devices, manager_ip, manager_port)
    td.barrier()

    client = start_cugraph_dask_client(rank, dask_scheduler_file)

    from distributed import Event as Dask_Event

    event = Dask_Event("cugraph_store_creation_event")
    download_event = Dask_Event("dataset_download_event")

    td.barrier()

    import cugraph
    from cugraph_pyg.data import DaskGraphStore
    from cugraph_pyg.loader import DaskNeighborLoader

    if rank == 0:
        print("Rank 0 downloading dataset")
        dataset = NodePropPredDataset(name="ogbn-mag")
        data = dataset[0]
        download_event.set()
        print("Dataset downloaded")
    else:
        if download_event.wait(timeout=1000):
            print(f"Rank {rank} loading dataset")
            dataset = NodePropPredDataset(name="ogbn-mag")
            data = dataset[0]
            print(f"Rank {rank} loaded dataset successfully")

    ei = data[0]["edge_index_dict"][("paper", "cites", "paper")]
    G = {
        ("paper", "cites", "paper"): np.stack(
            [np.concatenate([ei[0], ei[1]]), np.concatenate([ei[1], ei[0]])]
        )
    }
    N = {"paper": data[0]["num_nodes_dict"]["paper"]}

    fs = cugraph.gnn.FeatureStore(backend="torch")

    fs.add_data(
        torch.as_tensor(data[0]["node_feat_dict"]["paper"], device=features_device),
        "paper",
        "x",
    )

    fs.add_data(torch.as_tensor(data[1]["paper"].T[0], device=device_id), "paper", "y")

    num_papers = data[0]["num_nodes_dict"]["paper"]

    if rank == 0:
        train_perc = 0.1
        all_train_nodes = torch.randperm(num_papers)
        all_train_nodes = all_train_nodes[: int(train_perc * num_papers)]
        train_nodes = all_train_nodes[: int(len(all_train_nodes) / world_size)]

        train_mask = torch.full((num_papers,), -1, device=device_id)
        train_mask[train_nodes] = 1
        fs.add_data(train_mask, "paper", "train")

    print(f"Rank {rank} finished loading graph and feature data")

    if rank == 0:
        print("Rank 0 creating its cugraph store and initializing distributed graph")
        # Rank 0 will initialize the distributed cugraph graph.
        cugraph_store_create_start = time.perf_counter_ns()
        print("G:", G[("paper", "cites", "paper")].shape)
        cugraph_store = DaskGraphStore(fs, G, N, multi_gpu=True)
        cugraph_store_create_end = time.perf_counter_ns()
        print(
            "cuGraph Store created on rank 0 in "
            f"{(cugraph_store_create_end - cugraph_store_create_start) / 1e9:3.4f} s"
        )
        client.publish_dataset(train_nodes=all_train_nodes)
        event.set()
        print("Rank 0 done with cugraph store creation")
    else:
        if event.wait(timeout=1000):
            print(f"Rank {rank} creating cugraph store")
            train_nodes = client.get_dataset("train_nodes")
            train_nodes = train_nodes[
                int(rank * len(train_nodes) / world_size) : int(
                    (rank + 1) * len(train_nodes) / world_size
                )
            ]

            train_mask = torch.full((num_papers,), -1, device=device_id)
            train_mask[train_nodes] = 1
            fs.add_data(train_mask, "paper", "train")

            # Will automatically use the stored distributed cugraph graph on rank 0.
            cugraph_store_create_start = time.perf_counter_ns()
            cugraph_store = DaskGraphStore(fs, G, N, multi_gpu=True)
            cugraph_store_create_end = time.perf_counter_ns()
            print(
                f"Rank {rank} created cugraph store in "
                f"{(cugraph_store_create_end - cugraph_store_create_start) / 1e9:3.4f}"
                " s"
            )
            print(f"Rank {rank} done with cugraph store creation")

    end_time_preprocess = time.perf_counter_ns()
    print(f"rank {rank}: train {train_nodes.shape}", flush=True)
    print(
        f"rank {rank}: all preprocessing took"
        f" {(end_time_preprocess - start_time_preprocess) / 1e9:3.4f}",
        flush=True,
    )
    td.barrier()
    model = (
        CuGraphSAGE(in_channels=128, hidden_channels=64, out_channels=349, num_layers=3)
        .to(torch.float32)
        .to(device_id)
    )
    model = ddp(model, device_ids=[device_id], output_device=device_id)
    td.barrier()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        start_time_train = time.perf_counter_ns()
        model.train()

        start_time_loader = time.perf_counter_ns()
        cugraph_bulk_loader = DaskNeighborLoader(
            cugraph_store,
            train_nodes,
            batch_size=250,
            num_neighbors=[10, 10, 10],
            seeds_per_call=1000,
            batches_per_partition=2,
            replace=False,
        )
        end_time_loader = time.perf_counter_ns()
        total_time_loader = (end_time_loader - start_time_loader) / 1e9

        total_loss = 0
        num_batches = 0

        print(f"rank {rank} starting epoch {epoch}")
        with td.algorithms.join.Join([model]):
            total_time_sample = 0
            total_time_forward = 0
            total_time_backward = 0

            start_time_sample = time.perf_counter_ns()
            for iter_i, hetero_data in enumerate(cugraph_bulk_loader):
                end_time_sample = time.perf_counter_ns()
                total_time_sample += (end_time_sample - start_time_sample) / 1e9
                num_batches += 1

                if iter_i % 20 == 0:
                    print(f"iteration {iter_i}")

                # train
                train_mask = hetero_data.train_dict["paper"]
                y_true = hetero_data.y_dict["paper"]

                start_time_forward = time.perf_counter_ns()
                y_pred = model(
                    hetero_data.x_dict["paper"].to(device_id).to(torch.float32),
                    hetero_data.edge_index_dict[("paper", "cites", "paper")].to(
                        device_id
                    ),
                    (len(y_true), len(y_true)),
                )
                end_time_forward = time.perf_counter_ns()
                total_time_forward += (end_time_forward - start_time_forward) / 1e9

                y_true = F.one_hot(
                    y_true[train_mask].to(torch.int64), num_classes=349
                ).to(torch.float32)

                y_pred = y_pred[train_mask]

                loss = F.cross_entropy(y_pred, y_true)

                start_time_backward = time.perf_counter_ns()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                end_time_backward = time.perf_counter_ns()
                total_time_backward += (end_time_backward - start_time_backward) / 1e9

                total_loss += loss.item()

                del y_true
                del y_pred
                del loss
                del hetero_data
                gc.collect()

                start_time_sample = time.perf_counter_ns()

            end_time_train = time.perf_counter_ns()
            print(
                f"epoch {epoch} "
                f"total time: {(end_time_train - start_time_train) / 1e9:3.4f} s"
                f"\nloader create time per batch: {total_time_loader / num_batches} s"
                f"\nsampling/load time per batch: {total_time_sample / num_batches} s"
                f"\nforward time per batch: {total_time_forward / num_batches} s"
                f"\nbackward time per batch: {total_time_backward / num_batches} s"
                f"\nnum batches: {num_batches}"
            )
            print(f"loss after epoch {epoch}: {total_loss / num_batches}")

    td.barrier()
    if rank == 0:
        print("DONE", flush=True)
        client.unpublish_dataset("train_nodes")
        event.clear()

    td.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torch_devices",
        type=str,
        default="0,1",
        help="GPU to allocate to pytorch for model, graph data, and node label storage",
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
        "--features_on_gpu",
        type=bool,
        default=True,
        help="Whether to store the features on each worker's GPU",
        required=False,
    )

    parser.add_argument(
        "--torch_manager_ip",
        type=str,
        default="127.0.0.1",
        help="The torch distributed manager ip address",
        required=False,
    )

    parser.add_argument(
        "--torch_manager_port",
        type=str,
        default="12346",
        help="The torch distributed manager port",
        required=False,
    )

    parser.add_argument(
        "--dask_scheduler_file",
        type=str,
        help="The path to the dask scheduler file",
        required=False,
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.dask_scheduler_file is None:
        warnings.warn(
            "You must provide the dask scheduler file " "to run this example.  Exiting."
        )

    else:
        torch_devices = [int(d) for d in args.torch_devices.split(",")]

        train_args = (
            torch_devices,
            args.torch_manager_ip,
            args.torch_manager_port,
            args.dask_scheduler_file,
            args.num_epochs,
            args.features_on_gpu,
        )

        tmp.spawn(train, args=train_args, nprocs=len(torch_devices))


if __name__ == "__main__":
    main()
