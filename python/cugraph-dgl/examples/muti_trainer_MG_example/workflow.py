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

import dgl
import torch
import time
from distributed import Client, Event as Dask_Event
import tempfile


def enable_spilling():
    import cudf

    cudf.set_option("spill", True)


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
    client.run(enable_spilling)
    print("Dask Cluster Setup Complete")
    del client
    return cluster.scheduler_address


def create_dask_client(scheduler_address):
    from cugraph.dask.comms import comms as Comms

    client = Client(scheduler_address)
    Comms.initialize(p2p=True)
    return client


def initalize_pytorch_worker(dev_id):
    import cupy as cp
    import rmm

    dev = cp.cuda.Device(
        dev_id
    )  # Create cuda context on the right gpu, defaults to gpu-0
    dev.use()
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=10e9,
        maximum_pool_size=15e9,
        devices=[dev_id],
    )

    if dev_id == 0:
        torch.cuda.memory.change_current_allocator(rmm.rmm_torch_allocator)

    torch.cuda.set_device(dev_id)
    cp.cuda.set_allocator(rmm.rmm_cupy_allocator)
    enable_spilling()
    print("device_id", dev_id, flush=True)


def load_dgl_dataset(dataset_name="ogbn-products"):
    from ogb.nodeproppred import DglNodePropPredDataset

    dataset = DglNodePropPredDataset(name=dataset_name)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
    )
    g, label = dataset[0]
    g.ndata["label"] = label
    if len(g.etypes) <= 1:
        g = dgl.add_self_loop(g)
    else:
        for etype in g.etypes:
            if etype[0] == etype[2]:
                # only add self loops for src->dst
                g = dgl.add_self_loop(g, etype=etype)

    g = g.int()
    train_idx = train_idx.int()
    valid_idx = valid_idx.int()
    test_idx = test_idx.int()
    return g, train_idx, valid_idx, test_idx


def create_cugraph_graphstore_from_dgl_dataset(
    dataset_name="ogbn-products", single_gpu=False
):
    from cugraph_dgl import cugraph_storage_from_heterograph

    dgl_g, train_idx, valid_idx, test_idx = load_dgl_dataset(dataset_name)
    cugraph_gs = cugraph_storage_from_heterograph(dgl_g, single_gpu=single_gpu)
    return cugraph_gs, train_idx, valid_idx, test_idx


def create_dataloader(gs, train_idx, device):
    import cugraph_dgl

    temp_dir = tempfile.TemporaryDirectory()
    sampler = cugraph_dgl.dataloading.NeighborSampler([10, 20])
    dataloader = cugraph_dgl.dataloading.DataLoader(
        gs,
        train_idx,
        sampler,
        sampling_output_dir=temp_dir.name,
        batches_per_partition=10,
        device=device,  # Put the sampled MFGs on CPU or GPU
        use_ddp=True,  # Make it work with distributed data parallel
        batch_size=1024,
        shuffle=False,  # Whether to shuffle the nodes for every epoch
        drop_last=False,
        num_workers=0,
    )
    return dataloader


def run_workflow(rank, devices, scheduler_address):
    # Below sets gpu_num
    dev_id = devices[rank]
    initalize_pytorch_worker(dev_id)
    device = torch.device(f"cuda:{dev_id}")
    # cugraph dask client initialization
    client = create_dask_client(scheduler_address)

    # Pytorch training worker initialization
    dist_init_method = "tcp://{master_ip}:{master_port}".format(
        master_ip="127.0.0.1", master_port="12346"
    )

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=dist_init_method,
        world_size=len(devices),
        rank=rank,
    )

    print(f"rank {rank}.", flush=True)
    print("Initalized across GPUs.")

    event = Dask_Event("cugraph_gs_creation_event")
    if rank == 0:
        (
            gs,
            train_idx,
            valid_idx,
            test_idx,
        ) = create_cugraph_graphstore_from_dgl_dataset(
            "ogbn-products", single_gpu=False
        )
        client.publish_dataset(cugraph_gs=gs)
        client.publish_dataset(train_idx=train_idx)
        client.publish_dataset(valid_idx=valid_idx)
        client.publish_dataset(test_idx=test_idx)
        event.set()
    else:
        if event.wait(timeout=1000):
            gs = client.get_dataset("cugraph_gs")
            train_idx = client.get_dataset("train_idx")
            valid_idx = client.get_dataset("valid_idx")
            test_idx = client.get_dataset("test_idx")
        else:
            raise RuntimeError(f"Fetch cugraph_gs to worker_id {rank} failed")

    torch.distributed.barrier()
    print(f"Loading cugraph_store to worker {rank} is complete", flush=True)
    dataloader = create_dataloader(gs, train_idx, device)
    print("Data Loading Complete", flush=True)
    del gs  # Clean up gs reference
    # Comment below
    st = time.time()
    for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
        pass
    et = time.time()
    print(f"Warmup loading took = {et-st} s on worker = {rank}")
    torch.distributed.barrier()

    n_epochs = 30
    total_st = time.time()
    for i in range(0, n_epochs):
        st = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            pass
            # print(len(input_nodes))
            # print(len(seeds))
            # train_model()
        et = time.time()
        print(f"Data Loading took = {et-st} s for epoch = {i} on worker = {rank}")
    torch.distributed.barrier()
    total_et = time.time()
    print(
        f"Total time taken on n_epochs {n_epochs} = {total_et-total_st} s",
        f"measured by worker = {rank}",
    )

    # cleanup dask cluster
    if rank == 0:
        client.unpublish_dataset("cugraph_gs")
        client.unpublish_dataset("train_idx")
        client.unpublish_dataset("valid_idx")
        client.unpublish_dataset("test_idx")
        event.clear()


if __name__ == "__main__":
    dask_worker_devices = [5, 6]
    scheduler_address = setup_cluster(dask_worker_devices)

    trainer_devices = [0, 1, 2]
    import torch.multiprocessing as mp

    mp.spawn(
        run_workflow,
        args=(trainer_devices, scheduler_address),
        nprocs=len(trainer_devices),
    )
