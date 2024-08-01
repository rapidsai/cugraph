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

import dgl
import torch
import pandas
import time
import tempfile
import argparse
import json
import os
import warnings

from datetime import timedelta

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


def init_ddp_worker(global_rank, local_rank, world_size, cugraph_id):
    import rmm

    rmm.reinitialize(
        devices=local_rank,
        managed_memory=True,
        pool_allocator=True,
    )

    import cupy

    cupy.cuda.Device(local_rank).use()
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)

    from cugraph.testing.mg_utils import enable_spilling

    enable_spilling()

    torch.cuda.set_device(local_rank)

    cugraph_comms_init(
        rank=global_rank, world_size=world_size, uid=cugraph_id, device=local_rank
    )

    wm_init(global_rank, world_size, local_rank, torch.cuda.device_count())


def load_dgl_dataset(dataset_root="dataset", dataset_name="ogbn-products"):
    from ogb.nodeproppred import DglNodePropPredDataset

    dataset = DglNodePropPredDataset(root=dataset_root, name=dataset_name)
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
    idx = {
        "train": train_idx.int(),
        "valid": valid_idx.int(),
        "test": test_idx.int(),
    }

    return g, idx, dataset.num_classes


def partition_data(
    g, split_idx, num_classes, edge_path, feature_path, label_path, meta_path
):
    # Split and save edge index
    os.makedirs(
        edge_path,
        exist_ok=True,
    )
    src, dst = g.all_edges(form="uv", order="eid")
    edge_index = torch.stack([src, dst])
    for (r, e) in enumerate(torch.tensor_split(edge_index, world_size, dim=1)):
        rank_path = os.path.join(edge_path, f"rank={r}.pt")
        torch.save(
            e.clone(),
            rank_path,
        )

    # Split and save features
    os.makedirs(
        feature_path,
        exist_ok=True,
    )

    nix = torch.arange(g.num_nodes())
    ndata = pandas.DataFrame({k: g.ndata[k][nix] for k in g.ndata.keys()})
    for (r, f) in enumerate(torch.tensor_split(nix, world_size)):
        rank_path = os.path.join(feature_path, f"rank={r}_feat.parquet")
        ndata.iloc[f].to_parquet(rank_path)

    # Split and save labels
    os.makedirs(
        label_path,
        exist_ok=True,
    )
    for (d, i) in split_idx.items():
        i_parts = torch.tensor_split(i, world_size)
        for r, i_part in enumerate(i_parts):
            rank_path = os.path.join(label_path, f"rank={r}")
            os.makedirs(rank_path, exist_ok=True)
            torch.save(i_part, os.path.join(rank_path, f"{d}.pt"))

    # Save metadata
    meta = {
        "num_classes": int(num_classes),
        "num_nodes": int(g.num_nodes()),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def load_partitioned_data(rank, edge_path, feature_path, label_path, meta_path):
    g = cugraph_dgl.Graph(
        is_multi_gpu=True, ndata_storage="wholegraph", edata_storage="wholegraph"
    )

    # Load metadata
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Load labels
    split_idx = {}
    for split in ["train", "test", "valid"]:
        split_idx[split] = torch.load(
            os.path.join(label_path, f"rank={rank}", f"{split}.pt")
        )

    # Load features
    ndata_df = pandas.read_parquet(
        os.path.join(feature_path, f"rank={rank}_feat.parquet")
    )
    ndata = {col: torch.as_tensor(s.values) for col, s in ndata_df.items()}
    g.add_nodes(meta["num_nodes"], data=ndata)

    # Load edge index
    src, dst = torch.load(os.path.join(edge_path, f"rank={rank}.pt"))
    g.add_edges(src.cuda(), dst.cuda(), data=None)

    return g, split_idx, meta["num_classes"]


def create_dataloader(gs, train_idx, device, temp_dir, stage):
    import cugraph_dgl

    temp_path = os.path.join(temp_dir, f"{stage}_{device}")
    os.mkdir(temp_path)

    sampler = cugraph_dgl.dataloading.NeighborSampler(
        [10, 20],
        directory=temp_path,
        batches_per_partition=10,
    )
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


def run_workflow(
    global_rank, local_rank, world_size, g, split_idx, num_classes, temp_dir
):
    from model import Sage, train_model

    # Below sets gpu_number
    dev_id = local_rank
    device = torch.device(f"cuda:{dev_id}")

    dataloader = create_dataloader(g, split_idx["train"], device, temp_dir, "train")
    print("Dataloader Creation Complete", flush=True)
    num_feats = g.ndata["feat"].shape[1]
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
    train_model(model, g, opt, dataloader, n_epochs, global_rank, split_idx["valid"])
    torch.distributed.barrier()
    total_et = time.time()
    print(
        f"Total time taken on n_epochs {n_epochs} = {total_et-total_st} s",
        f"measured by worker = {global_rank}",
    )

    wm_finalize()
    cugraph_comms_shutdown()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_root", type=str, default="dataset")
        parser.add_argument("--tempdir_root", type=str, default=None)
        parser.add_argument("--dataset", type=str, default="ogbn-products")
        parser.add_argument("--skip_partition", action="store_true")
        args = parser.parse_args()

        torch.distributed.init_process_group(
            "nccl",
            timeout=timedelta(minutes=60),
        )
        world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(local_rank)

        # Create the uid needed for cuGraph comms
        if global_rank == 0:
            cugraph_id = [cugraph_comms_create_unique_id()]
        else:
            cugraph_id = [None]
        torch.distributed.broadcast_object_list(cugraph_id, src=0, device=device)
        cugraph_id = cugraph_id[0]

        init_ddp_worker(global_rank, local_rank, world_size, cugraph_id)

        # Split the data
        edge_path = os.path.join(args.dataset_root, args.dataset + "_eix_part")
        feature_path = os.path.join(args.dataset_root, args.dataset + "_fea_part")
        label_path = os.path.join(args.dataset_root, args.dataset + "_label_part")
        meta_path = os.path.join(args.dataset_root, args.dataset + "_meta.json")

        if not args.skip_partition and global_rank == 0:
            partition_data(*load_dgl_dataset(args.dataset_root, args.dataset), edge_path, feature_path, label_path, meta_path)
        torch.distributed.barrier()

        print("loading partitions...")
        g, split_idx, num_classes = load_partitioned_data(
            rank=global_rank,
            edge_path=edge_path,
            feature_path=feature_path,
            label_path=label_path,
            meta_path=meta_path,
        )
        print(f"rank {global_rank} has loaded its partition")
        torch.distributed.barrier()

        with tempfile.TemporaryDirectory(dir=args.tempdir_root) as directory:
            run_workflow(
                global_rank,
                local_rank,
                world_size,
                g,
                split_idx,
                num_classes,
                directory,
            )
    else:
        warnings.warn("This script should be run with 'torchrun`.  Exiting.")
