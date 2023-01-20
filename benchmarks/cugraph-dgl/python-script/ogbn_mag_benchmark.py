# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import torch as th
import torch.nn.functional as F
import time
import argparse
## DGL Specific Import
from cugraph_dgl import c

def load_dgl_graph():
    from ogb.nodeproppred import DglNodePropPredDataset

    dataset = DglNodePropPredDataset(name="ogbn-mag", root='/datasets/vjawa/gnn/')
    split_idx = dataset.get_idx_split()
    g, labels = dataset[0]        
    # Uncomment for obgn-mag
    labels = labels["paper"].flatten()
    # labels = labels
    # transform = Compose([ToSimple(), AddReverse()])
    # g = transform(g)
    return g, labels, dataset.num_classes, split_idx


def sampling_func(g, seed_nodes,labels, train_loader):
    category = "paper"
    for input_nodes, seeds, blocks in train_loader:
        seeds = seeds[category]
        feat = blocks[0].srcdata['feat']['paper']
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark Sampling')
    parser.add_argument('--batch_size', type=int, default=100_000)
    parser.add_argument('--use_cugraph',dest='use_cugraph', action='store_true', default=True)
    parser.add_argument('--use_dgl_upstream', dest='use_cugraph', action='store_false')
    parser.add_argument('--single_gpu', dest='single_gpu', action='store_true', default=True)
    parser.add_argument('--multi_gpu', dest='single_gpu', action='store_false')
    parser.add_argument('--n_gpus', type=int, default=1)
    
    args = parser.parse_args()
    print(args, flush=True)
            
    single_gpu = args.single_gpu
    use_cugraph = args.use_cugraph
    batch_size = args.batch_size
    n_gpus = args.n_gpus

    if single_gpu:
        import rmm
        rmm.reinitialize(pool_allocator=True,initial_pool_size=5e+9, maximum_pool_size=22e+9)
    else:
        #### Dask Cluster
        from dask_cuda import LocalCUDACluster
        from cugraph.dask.comms import comms as Comms
        from dask.distributed import Client

        #Change according to your GPUS
        #Client at GPU-0
        #Workers at specifed GPUS
        #UCX seems to be freezing :-0 on DGX
        cuda_visible_devices = ','.join([str(i) for i in range(1,n_gpus+1)])
        cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=cuda_visible_devices, protocol='tcp', rmm_pool_size='12 GB', 
                                  jit_unspill=True)
        client = Client(cluster)
        Comms.initialize(p2p=True)
        
    
    device = 'cuda'
    g, labels, num_classes, split_idx = load_dgl_graph()
    g = g.to(device)
    
    if use_cugraph:
        if not single_gpu:
            g = g.int()
        g = cugraph_storage_from_heterograph(g, single_gpu=single_gpu)
    
    
    indx_type = g.idtype
    subset_split_idx = {'train': {k: v.to(device).to(g.idtype) for k,v in split_idx['train'].items()},
                       'valid' : {k: v.to(device).to(g.idtype) for k,v in split_idx['valid'].items()},
                        'test' : {k: v.to(device).to(g.idtype) for k,v in split_idx['test'].items()},
                       }


    sampler = dgl.dataloading.MultiLayerNeighborSampler([20,25], prefetch_node_feats={'paper':['feat']})
    train_loader = dgl.dataloading.DataLoader(
        g,
        subset_split_idx["train"],
        sampler,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        device=device,
    )

    ### Warmup RUN
    sampling_func(g, subset_split_idx['train'],labels, train_loader)
    
    ### Benchmarking RUN
    st = time.time()
    sampling_func(g, subset_split_idx['train'],labels, train_loader)
    et = time.time()    
    print(f"Sampling time taken  = {et-st} s")