# cugraph_dgl

[RAPIDS](https://rapids.ai) cugraph_dgl enables the ability to use cugraph Property Graphs with DGL.  This cugraph backend allows DGL users access to a collection of GPU-accelerated algorithms for graph analytics, such as sampling, centrality computation, and community detection.


The goal of `cugraph_dgl` is to enable Multi-Node Multi-GPU cugraph accelerated graphs to help train large-scale Graph Neural Networks(GNN) on DGL by providing a duck-typed version of the [DGLGraph](https://docs.dgl.ai/api/python/dgl.DGLGraph.html#dgl.DGLGraph)  which uses cugraph for storing graph structure and node/edge feature data. 

## Usage
```diff

+from cugraph_dgl.convert import cugraph_storage_from_heterograph
+cugraph_g = cugraph_storage_from_heterograph(dgl_g)

sampler = dgl.dataloading.NeighborSampler(
        [15, 10, 5], prefetch_node_feats=['feat'], prefetch_labels=['label'])

train_dataloader = dgl.dataloading.DataLoader(
- dgl_g, 
+ cugraph_g,
train_idx, 
sampler, 
device=device, 
batch_size=1024,
shuffle=True,
drop_last=False, 
num_workers=0)
```

