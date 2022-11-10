# cugraph_dgl

## Description

[RAPIDS](https://rapids.ai) cugraph_dgl provides a duck-typed version of the DGLHeteroGraph class made for cuGraph
    for storing graph structure and node/edge feature data. Using cugraph backend allows users to use GPU accelerated graph algorithms like sampling , traversal etc. 

## Conda
TODO

## Build from Source

### Create the conda development environment
```
mamba env create -n cugraph_dgl_dev --file conda/cugraph_dgl_dev_11.6.yml
```

### Install  in editable model
```
pip install -e . 
```

### Run tests

```
pytest tests/*
```


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


