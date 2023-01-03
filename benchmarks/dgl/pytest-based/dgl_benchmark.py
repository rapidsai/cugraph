# Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

import pandas as pd
import os
import pytest_benchmark
import pytest
import torch
import dgl

def get_edgelist(scale, edgefactor, dataset_dir = '../datasets'):
    fp = os.path.join(dataset_dir, f'rmat_scale_{scale}_edgefactor_{edgefactor}.parquet')
    return pd.read_parquet(fp)
                      
def create_dgl_graph_from_df(df):
    src_tensor = torch.as_tensor(df['src'].values)
    dst_tensor = torch.as_tensor(df['dst'].values)
    # Reverse edges to match cuGraph behavior
    g = dgl.graph(data = (dst_tensor, src_tensor))
    return g



@pytest.mark.parametrize("scale_edge_factor", [[24,16],[25,16]])
@pytest.mark.parametrize("batch_size", [100, 500, 1_000, 2_500, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000])
@pytest.mark.parametrize("fanout", [[10, 25]])
def bench_dgl_pure_gpu(benchmark, scale_edge_factor, batch_size, fanout):
    df = get_edgelist(scale_edge_factor[0], scale_edge_factor[1])
    g = create_dgl_graph_from_df(df).to('cuda')
    assert g.device.type =='cuda'
    seed_nodes = torch.as_tensor(df['dst'][:batch_size])
    seed_nodes = seed_nodes.to('cuda')
    assert len(seed_nodes)==batch_size
    ### Reverse because dgl sampler samples from destination to source
    fanout.reverse()
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
    
    input_nodes, output_nodes, blocks = benchmark(
        sampler.sample,
        g,
        seed_nodes=seed_nodes
    )
    assert len(output_nodes)==batch_size
    return


@pytest.fixture
def dgl_graph_26_16():
    df = get_edgelist(26, 16)
    g = create_dgl_graph_from_df(df)
    return g, df


@pytest.mark.parametrize("batch_size", [100, 500, 1_000, 2_500, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000])
@pytest.mark.parametrize("fanout", [[10, 25]])
def bench_dgl_uva(benchmark, dgl_graph_26_16, batch_size, fanout):
    g,df = dgl_graph_26_16
    assert g.device.type =='cpu'
    seed_nodes = torch.as_tensor(df['dst'][:30_000_000])
    seed_nodes = seed_nodes.to('cuda')
    ### Reverse because dgl sampler samples from destination to source
    fanout.reverse()
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
    dataloader = dgl.dataloading.DataLoader(
    g,                               
    seed_nodes,                        # train_nid must be on GPU.
    sampler,
    device=torch.device('cuda:0'),    # The device argument must be GPU.
    num_workers=0,                    # Number of workers must be 0.
    use_uva=True,
    batch_size=batch_size,
    drop_last=False,
    shuffle=False)

    def uva_benchmark(dataloader_it):
        input_nodes, output_nodes, blocks = next(dataloader_it)
        return  input_nodes, output_nodes, blocks 

    # added iterations and rounds to prevent dataloader going over num batches
    dataloader_it = iter(dataloader)
    input_nodes, output_nodes, blocks = benchmark.pedantic(uva_benchmark,kwargs = {'dataloader_it': dataloader_it}, iterations=10, rounds=10)
    assert len(output_nodes)==batch_size
    return
