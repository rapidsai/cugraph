# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuGraph - GPU Graph Analytics</div>

The [RAPIDS](https://rapids.ai) cuGraph library is a collection of graph analytics that process data found in GPU Dataframes - see [cuDF](https://github.com/rapidsai/cudf).  cuGraph aims to provide a NetworkX-like API that will be familiar to data scientists, so they can now build GPU-accelerated workflows more easily.

 For more project details, see [rapids.ai](https://rapids.ai/).

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/cudf/blob/master/README.md) ensure you are on the latest branch.



```markdown
import cugraph

# assuming that data has been loaded into a cuDF (using read_csv) Dataframe
gdf = cudf.read_csv("graph_data.csv", names=["src", "dst"], dtype=["int32", "int32"] )

# create a Graph using the source (src) and destination (dst) vertex pairs the GDF  
G = cugraph.Graph()
G.add_edge_list(gdf, source='src', destination='dst')

# Call cugraph.pagerank to get the pagerank scores
gdf_page = cugraph.pagerank(G)

for i in range(len(gdf_page)):
	print("vertex " + str(gdf_page['vertex'][i]) + 
		" PageRank is " + str(gdf_page['pagerank'][i]))  
```



## Supported Algorithms:

| Algorithm                                     | Scale      | Notes                        |
| :-------------------------------------------- | ---------- | ---------------------------- |
| PageRank                                      | Multi-GPU  |                              |
| Personal PageRank                             | Single-GPU |                              |
| Katz Centrality                               | Single-GPU |                              |
| Jaccard Similarity                            | Single-GPU |                              |
| Weighted Jaccard                              | Single-GPU |                              |
| Overlap Similarity                            | Single-GPU |                              |
| SSSP                                          | Single-GPU | Updated to provide path info |
| BFS                                           | Single-GPU | Also BSP version             |
| Triangle Counting                             | Single-GPU |                              |
| K-Core                                        | Single-GPU |                              |
| Core Number                                   | Single-GPU |                              |
| Subgraph Extraction                           | Single-GPU |                              |
| Spectral Clustering - Balanced-Cut            | Single-GPU |                              |
| Spectral Clustering - Modularity Maximization | Single-GPU |                              |
| Louvain                                       | Single-GPU |                              |
| Renumbering                                   | Single-GPU |                              |
| Basic Graph Statistics                        | Single-GPU |                              |
| Weakly Connected Components                   | Single-GPU |                              |
| Strongly Connected Components                 | Single-GPU |                              |





## cuGraph Notice

The current version of cuGraph has some limitations:

- Vertex IDs need to be 32-bit integers.
- Vertex IDs are expected to be contiguous integers starting from 0.

cuGraph provides the renumber function to mitigate this problem. Input vertex IDs for the renumber function can be either 32-bit or 64-bit integers, can be non-contiguous, and can start from an arbitrary number. The renumber function maps the provided input vertex IDs to 32-bit contiguous integers starting from 0. cuGraph still requires the renumbered vertex IDs to be representable in 32-bit integers. These limitations are being addressed and will be fixed soon.

Release 0.11 includes a new 'Graph' class that could cause errors to existing code.  Please see the [Trainsition Guide](TRANSITIONGUIDE.md)



## Getting cuGraph
### Intro
There are 3 ways to get cuGraph :
1. [Quick start with Docker Demo Repo](#quick)
1. [Conda Installation](#conda)
1. [Build from Source](#source)



<a name="quick"></a>

## Quick Start

Please see the [Demo Docker Repository](https://hub.docker.com/r/rapidsai/rapidsai/), choosing a tag based on the NVIDIA CUDA version you’re running. This provides a ready to run Docker container with example notebooks and data, showcasing how you can utilize all of the RAPIDS libraries: cuDF, cuML, and cuGraph.



<a name="conda"></a>

### Conda

It is easy to install cuGraph using conda. You can get a minimal conda installation with [Miniconda](https://conda.io/miniconda.html) or get the full installation with [Anaconda](https://www.anaconda.com/download).

Install and update cuGraph using the conda command:

```bash
# CUDA 9.2
conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cugraph cudatoolkit=9.2

# CUDA 10.0
conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cugraph cudatoolkit=10.0

# CUDA 10.1
conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cugraph cudatoolkit=10.1
```

Note: This conda installation only applies to Linux and Python versions 3.6/3.7.



<a name="source"></a>

### Build from Source and Contributing

Please see our [guide for building and contributing to cuGraph](CONTRIBUTING.md).



## Documentation

Python API documentation can be generated from [docs](docs) directory.







------



## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="80%"/></p>

### Apache Arrow on GPU

The GPU version of [Apache Arrow](https://arrow.apache.org/) is a common API that enables efficient interchange of tabular data between processes running on the GPU. End-to-end computation on the GPU avoids unnecessary copying and converting of data off the GPU, reducing compute time and cost for high-performance analytics common in artificial intelligence workloads. As the name implies, cuDF uses the Apache Arrow columnar data format on the GPU. Currently, a subset of the features in Apache Arrow are supported.
