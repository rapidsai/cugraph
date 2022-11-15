<h1 align="center";>
  <br>
  <img src="../img/cugraph_logo_2.png" alt="cuGraph" width="300">
</h1>

GNN Support

Rapids offers support to GNN (Graph Neural Networks). Several components of the Rapids ecosystem fit into a typical GNN framework as shown below.

<h1 align="center";>
  <img src="../img/gnn_framework.png" alt="cuGraph" width="500">
</h1>

Rapids also has elements specifically geared to GNN's. Due to the degree distribution of nodes, memory bottlenecks are the pain point for large scale graphs. To solve this problem, sampling operations form the backbone for Graph Neural Networks (GNN) training. However, current sampling methods provided by other libraries are not optimized enough for the whole process of GNN training, the main limit to performance is moving data between the hosts and devices. In cuGraph, we provide an end-to-end solution from data loading to training all on the GPUs.

CuGraph now supports compatibility with DGL and PyTorch Geometric (PyG) by allowing conversion between a cuGraph object and a DGL or PyG object, making it possible for DGL and PyG users to access efficient data loader and graph operations (such as uniformed sampling) implementations in cuGraph, as well as keep their models unchanged in DGL or PyG. We achieved XXX times speedup compare with the original implementation in DGL and PyG.

