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
import tempfile
import argparse
import os

import cugraph_dgl

from cugraph.gnn import (
    cugraph_comms_init,
    cugraph_comms_shutdown,
    cugraph_comms_create_unique_id,
)

from pylibwholegraph.torch.initialize import (
    init as wm_init,
    finalize as wm_finalize,
)

# Allow computation on objects that are larger than GPU memory
# https://docs.rapids.ai/api/cudf/stable/developer_guide/library_design/#spilling-to-host-memory
os.environ["CUDF_SPILL"] = "1"


def initalize_pytorch_worker(dev_id):
    import cupy as cp
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

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

    from cugraph.testing.mg_utils import enable_spilling
    enable_spilling()

    torch.cuda.set_device(dev_id)
    cp.cuda.set_allocator(rmm_cupy_allocator)
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
    return g, train_idx, valid_idx, test_idx, dataset.num_classes


def create_cugraph_graphstore_from_dgl_dataset(
    dataset, rank, world_size
):
    (g, train_idx, valid_idx, test_idx, num_classes) = dataset
    # Partition the data
    cg = cugraph_dgl.Graph(is_multi_gpu=True,  ndata_storage='wholegraph', edata_storage='wholegraph')
    
    nix = torch.tensor_split(torch.arange(g.num_nodes()), world_size)[rank]
    ndata = {
        k: g.ndata[k][nix].cuda()
        for k in g.ndata.keys()
    }

    eix = torch.tensor_split(torch.arange(g.num_edges()), world_size)[rank]
    src, dst = g.all_edges(form='uv', order='eid')
    edata = {
        k: g.edata[k][eix].cuda()
        for k in g.edata.keys()
    }

    cg.add_nodes(g.num_nodes(), data=ndata)
    cg.add_edges(
        torch.tensor_split(src, world_size)[rank].cuda(),
        torch.tensor_split(dst, world_size)[rank].cuda(),
        data=edata,
    )

    return (cg, torch.tensor_split(train_idx, world_size)[rank].to(torch.int64), torch.tensor_split(valid_idx, world_size)[rank].to(torch.int64), torch.tensor_split(test_idx, world_size)[rank].to(torch.int64), num_classes)


def create_dataloader(gs, train_idx, device, temp_dir, stage):
    import cugraph_dgl

    temp_path = os.path.join(temp_dir, f'{stage}_{device}')
    os.mkdir(temp_path)

    sampler = cugraph_dgl.dataloading.NeighborSampler([10, 20], directory=temp_path, batches_per_partition=10,)
    dataloader = cugraph_dgl.dataloading.FutureDataLoader(
        gs,
        train_idx,
        sampler,
        device=device,  # Put the sampled MFGs on CPU or GPU
        use_ddp=True,  # Make it work with distributed data parallel
        batch_size=1024,
        shuffle=False,  # Whether to shuffle the nodes for every epoch
        drop_last=False,
        num_workers=0,
    )
    return dataloader


def run_workflow(rank, world_size, cugraph_id, dataset, temp_dir):
    from model import Sage, train_model

    # Below sets gpu_number
    dev_id = rank
    initalize_pytorch_worker(dev_id)
    device = torch.device(f"cuda:{dev_id}")

    # Pytorch training worker initialization
    dist_init_method = "tcp://{master_ip}:{master_port}".format(
        master_ip="127.0.0.1", master_port="12346"
    )

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=dist_init_method,
        world_size=world_size,
        rank=rank,
    )

    cugraph_comms_init(rank=rank, world_size=world_size, uid=cugraph_id, device=rank)
    wm_init(rank, world_size, rank, world_size)

    print(f"rank {rank}.", flush=True)
    print("Initalized across GPUs.")

    (
        gs,
        train_idx,
        valid_idx,
        test_idx,
        num_classes,
    ) = create_cugraph_graphstore_from_dgl_dataset(
        dataset, rank, world_size,
    )
    del dataset

    torch.distributed.barrier()
    print(f"Loading graph to worker {rank} is complete", flush=True)

    dataloader = create_dataloader(gs, train_idx, device, temp_dir, 'train')
    print("Dataloader Creation Complete", flush=True)
    num_feats = gs.ndata["feat"].shape[1]
    hid_size = 256
    # Load Training example
    model = Sage(num_feats, hid_size, num_classes).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device,
    )
    torch.distributed.barrier()
    n_epochs = 10
    total_st = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    train_model(model, gs, opt, dataloader, n_epochs, rank, valid_idx)
    torch.distributed.barrier()
    total_et = time.time()
    print(
        f"Total time taken on n_epochs {n_epochs} = {total_et-total_st} s",
        f"measured by worker = {rank}",
    )

    wm_finalize()
    cugraph_comms_shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbn-products')
    args = parser.parse_args()

    from rmm.allocators.torch import rmm_torch_allocator
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

    # Create the uid needed for cuGraph comms
    cugraph_id = cugraph_comms_create_unique_id()

    ds = load_dgl_dataset(args.dataset)

    world_size = torch.cuda.device_count()

    with tempfile.TemporaryDirectory() as directory:
        torch.multiprocessing.spawn(
            run_workflow,
            args=(world_size, cugraph_id, ds, directory),
            nprocs=world_size,
        )
