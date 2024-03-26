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
# and PyTorch DDP to run a multi-GPU workflow.  Most users of the
# GNN packages will not interact with cuGraph directly.  This example
# is intented for users who want to extend cuGraph within a DDP workflow.

import os

import pandas
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
)

from pylibcugraph import MGGraph, ResourceHandle, GraphProperties, degrees

from ogb.nodeproppred import NodePropPredDataset

def init_pytorch(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

def calc_degree(rank:int, world_size: int, uid, edgelist):
    init_pytorch(rank, world_size)

    device = rank
    cugraph_comms_init(rank, world_size, uid, device)

    print(f'rank {rank} initialized cugraph')
    
    src = cudf.Series(
        np.array_split(edgelist[0], world_size)[rank]
    )
    dst = cudf.Series(
        np.array_split(edgelist[1], world_size)[rank]
    )
    
    seeds = cudf.Series(
        np.arange(rank * 50, (rank + 1) * 50)
    )
    handle = ResourceHandle(cugraph_comms_get_raft_handle().getHandle())

    print('constructing graph')
    G = MGGraph(
        handle,
        GraphProperties(is_multigraph=True,is_symmetric=False),
        [src],
        [dst],
    )
    print('graph constructed')

    print('calculating degrees')
    vertices, in_deg, out_deg = degrees(
        handle,
        G,
        seeds,
        do_expensive_check=False
    )
    print('degrees calculated')

    print('constructing dataframe')
    df = pandas.DataFrame({
        'v': vertices.get(),
        'in': in_deg.get(),
        'out': out_deg.get()
    })
    print(df)

    dist.barrier()
    cugraph_comms_shutdown()
    print(f'rank {rank} shut down cugraph')


def main():
    world_size = torch.cuda.device_count()
    uid = cugraph_comms_create_unique_id()

    dataset = NodePropPredDataset('ogbn-products')
    el = dataset[0][0]['edge_index'].astype('int64')

    tmp.spawn(
        calc_degree,
        args=(world_size, uid, el),
        nprocs=world_size,
    )

if __name__ == '__main__':
    main()