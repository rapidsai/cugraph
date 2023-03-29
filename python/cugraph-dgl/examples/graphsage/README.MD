Inductive Representation Learning on Large Graphs (GraphSAGE)
============

- Paper link: [http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)
- Author's code repo: [https://github.com/williamleif/graphsage-simple](https://github.com/williamleif/graphsage-simple)

For advanced usages, including training with multi-gpu/multi-node, and PyTorch Lightning, etc., more examples can be found in [advanced](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage/advanced) and [dist](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage/dist) directory.

Requirements
------------

```bash
mamba install ogb torchmetrics -c conda-forge
```

How to run
-------


### Minibatch training for node classification

Train w/ mini-batch sampling with cugraph_storage backend for node classification on "ogbn-products"

```bash
python3 node_classification.py --mode=gpu_cugraph_dgl
```
