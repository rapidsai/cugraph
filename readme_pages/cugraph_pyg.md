# cugraph_pyg

[RAPIDS](https://rapids.ai) cugraph_pyg enables the ability to use cugraph Property Graphs with PyTorch Geometric (PyG).  PyG users will have access to cuGraph and cuGraph-Service through the PyG GraphStore, FeatureStore, and Sampler interfaces.  Through cugraph_pyg, PyG users have the full power of cuGraph's GPU-accelerated algorithms for graph analytics, such as sampling, centrality computation, and community detection.


The goal of `cugraph_pyg` is to enable accelerated single-GPU and multi-node, multi-GPU cugraph accelerated graphs to help train large-scale Graph Neural Networks (GNN) on PyG by providing duck-typed drop-in replacements of the `GraphStore`, `FeatureStore`, and `Sampler` interfaces backed by either cuGraph or cuGraph-Service.

Users of cugraph_pyg have the option of installing either the cugraph or cugraph_service_client packages.  Only one is required.

## Usage
```
G = cuGraph.PropertyGraph()
...
feature_store, graph_store = to_pyg(G)
sampler = CuGraphSampler(
    data=(feature_store, graph_store),
    shuffle=True,
    num_neighbors=[10,25],
    batch_size=50,
)
...
```
