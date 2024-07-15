# Copyright (c) 2024, NVIDIA CORPORATION.
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

# This example shows how to use cuGraph nccl-only comms, pylibcuGraph,
# and PyTorch DDP to run a multi-GPU sampling workflow.  Most users of the
# GNN packages will not interact with cuGraph directly.  This example
# is intented for users who want to extend cuGraph within a DDP workflow.

import os
import re
import tempfile

import numpy as np
import torch
import torch.multiprocessing as tmp
import torch.distributed as dist

import cudf

from cugraph.gnn import (
    cugraph_comms_init,
    cugraph_comms_shutdown,
    cugraph_comms_create_unique_id,
    cugraph_comms_get_raft_handle,
    DistSampleWriter,
    UniformNeighborSampler,
)

from pylibcugraph import MGGraph, ResourceHandle, GraphProperties

from ogb.nodeproppred import NodePropPredDataset


def init_pytorch(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def sample(rank: int, world_size: int, uid, edgelist, directory):
    init_pytorch(rank, world_size)

    device = rank
    cugraph_comms_init(rank, world_size, uid, device)

    print(f"rank {rank} initialized cugraph")

    src = cudf.Series(np.array_split(edgelist[0], world_size)[rank])
    dst = cudf.Series(np.array_split(edgelist[1], world_size)[rank])

    seeds_per_rank = 50
    seeds = cudf.Series(np.arange(rank * seeds_per_rank, (rank + 1) * seeds_per_rank))
    handle = ResourceHandle(cugraph_comms_get_raft_handle().getHandle())

    print("constructing graph")
    G = MGGraph(
        handle,
        GraphProperties(is_multigraph=True, is_symmetric=False),
        [src],
        [dst],
    )
    print("graph constructed")

    sample_writer = DistSampleWriter(directory=directory, batches_per_partition=2)
    sampler = UniformNeighborSampler(
        G,
        sample_writer,
        fanout=[5, 5],
    )

    sampler.sample_from_nodes(seeds, batch_size=16, random_state=62)

    dist.barrier()
    cugraph_comms_shutdown()
    print(f"rank {rank} shut down cugraph")


def main():
    world_size = torch.cuda.device_count()
    uid = cugraph_comms_create_unique_id()

    dataset = NodePropPredDataset("ogbn-products")
    el = dataset[0][0]["edge_index"].astype("int64")

    with tempfile.TemporaryDirectory() as directory:
        tmp.spawn(
            sample,
            args=(world_size, uid, el, directory),
            nprocs=world_size,
        )

        print("Printing samples...")
        for file in os.listdir(directory):
            m = re.match(r"batch=([0-9]+)\.([0-9]+)\-([0-9]+)\.([0-9]+)\.parquet", file)
            rank, start, _, end = int(m[1]), int(m[2]), int(m[3]), int(m[4])
            print(f"File: {file} (batches {start} to {end} for rank {rank})")
            print(cudf.read_parquet(os.path.join(directory, file)))
            print("\n")


if __name__ == "__main__":
    main()
