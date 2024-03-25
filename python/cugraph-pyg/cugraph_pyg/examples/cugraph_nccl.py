import os

from time import sleep

import pandas
import numpy as np
import torch
import torch.multiprocessing as tmp
import torch.distributed as dist

from cugraph.gnn import (
    cugraph_comms_init,
    cugraph_comms_shutdown,
    cugraph_comms_create_unique_id,
    cugraph_comms_get_raft_handle,
)

from pylibcugraph import MGGraph, ResourceHandle, GraphProperties, degrees

def init_pytorch(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

def train(rank:int, world_size: int, uid, edgelist):
    init_pytorch(rank, world_size)

    device = rank
    cugraph_comms_init(rank, world_size, uid, device)

    import rmm
    rmm.reinitialize(pool_allocator=False, managed_memory=False, devices=[device])
    
    from rmm.allocators.cupy import rmm_cupy_allocator
    import cupy
    cupy.cuda.set_allocator(rmm_cupy_allocator)

    import cudf

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
        src_array=[src],
        dst_array=[dst],
        weight_array=None,
        edge_id_array=None,
        edge_type_array=None,
        num_arrays=1,
        store_transposed=False,
        do_expensive_check=False,
        drop_multi_edges=False,
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

    from ogb.nodeproppred import NodePropPredDataset
    dataset = NodePropPredDataset('ogbn-products')
    el = dataset[0][0]['edge_index'].astype('int64')

    tmp.spawn(
        train,
        args=(world_size, uid, el),
        nprocs=world_size,
    )

if __name__ == '__main__':
    main()