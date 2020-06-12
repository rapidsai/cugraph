# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuGraph - GPU Graph Analytics</div>

[![Build Status](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cugraph/job/branches/job/cugraph-branch-pipeline/badge/icon)](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cugraph/job/branches/job/cugraph-branch-pipeline/)

The [RAPIDS](https://rapids.ai) cuGraph library is a collection of GPU accelerated graph algorithms that process data found in [GPU DataFrames](https://github.com/rapidsai/cudf).  The vision of cuGraph is _to make graph analysis ubiquitous to the point that users just think in terms of analysis and not technologies or frameworks_.  To realize that vision, cuGraph operates, at the Python layer, on GPU DataFrames, allowing for seamless passing of data between ETL tasks in [cuDF](https://github.com/rapidsai/cudf) and machine learning tasks in [cuML](https://github.com/rapidsai/cuml).  Data scientists familiar with Python will quickly pick up how cuGraph integrates with the Pandas-like API of cuDF.  Likewise, users familiar with NetworkX will quickly recognize the NetworkX-like API provided in cuGraph, with the goal to allow existing code to be ported with minimal effort into RAPIDS.  For users familiar with C++/CUDA and graph structures, a C++ API is also provided.  However, there is less type and structure checking at the C++ layer.

 For more project details, see [rapids.ai](https://rapids.ai/).

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/cudf/blob/master/README.md) ensure you are on the latest branch.



```python
import cugraph

# read data into a cuDF DataFrame using read_csv
cu_M = cudf.read_csv("graph_data.csv", names=["src", "dst"], dtype=["int32", "int32"])

# We now have data as edge pairs
# create a Graph using the source (src) and destination (dst) vertex pairs
G = cugraph.Graph()
G.from_cudf_edgelist(cu_M, source='src', destination='dst')

# Let's now get the PageRank score of each vertex by calling cugraph.pagerank
df_page = cugraph.pagerank(G)

# Let's look at the PageRank Score (only do this on small graphs)
for i in range(len(df_page)):
	print("vertex " + str(df_page['vertex'].iloc[i]) +
		" PageRank is " + str(df_page['pagerank'].iloc[i]))
```


## Supported Algorithms

| Category     | Algorithm                              | Scale        |  Notes
| ------------ | -------------------------------------- | ------------ | ------------------- |
| Centrality   |                                        |              |                     |
|              | Katz                                   | Single-GPU   |                     |
|              | Betweenness Centrality                 | Single-GPU   |                     |
|              | Edge Betweenness Centrality            | Single-GPU   |                     |
| Community    |                                        |              |                     |
|              | Louvain                                | Single-GPU   |                     |
|              | Ensemble Clustering for Graphs         | Single-GPU   |                     |
|              | Spectral-Clustering - Balanced Cut     | Single-GPU   |                     |
|              | Spectral-Clustering                    | Single-GPU   |                     |
|              | Subgraph Extraction                    | Single-GPU   |                     |
|              | Triangle Counting                      | Single-GPU   |                     |
| Components   |                                        |              |                     |
|              | Weakly Connected Components            | Single-GPU   |                     |
|              | Strongly Connected Components          | Single-GPU   |                     |
| Core         |                                        |              |                     |
|              | K-Core                                 | Single-GPU   |                     |
|              | Core Number                            | Single-GPU   |                     |
|              | K-Truss                                | Single-GPU   |                     |
| Layout       |                                        |              |                     |
|              | Force Atlas 2                          | Single-GPU   |                     |
| Link Analysis|                                        |              |                     |
|              | Pagerank                               | Single-GPU   |                     |
|              | Personal Pagerank                      | Single-GPU   |                     |
| Link Prediction |                                     |              |                     |
|              | Jacard Similarity                      | Single-GPU   |                     |
|              | Weighted Jacard Similarity             | Single-GPU   |                     |
|              | Overlap Similarity                     | Single-GPU   |                     |
| Traversal    |                                        |              |                     |
|              | Breadth First Search (BFS)             | Single-GPU   |                     |
|              | Single Source Shortest Path (SSSP)     | Single-GPU   |                     |
| Structure    |                                        |              |                     |
|              | Renumbering                            | Single-GPU   | Also for multiple columns  |
|              | Symmetrize                             | Single-GPU   |                     |

## Supported Graph
| Type            |  Description                                        |
| --------------- | --------------------------------------------------- |
| Graph           | An undirected Graph                                 |
| DiGraph         | A Directed Graph                                    |


## cuGraph Notice
The current version of cuGraph has some limitations:

- Vertex IDs need to be 32-bit integers.
- Vertex IDs are expected to be contiguous integers starting from 0.
--  If the starting index is not zero, cuGraph will add disconnected vertices to fill in the missing range.  (Auto-) Renumbering fixes this issue

cuGraph provides the renumber function to mitigate this problem. Input vertex IDs for the renumber function can be any type, can be non-contiguous, and can start from an arbitrary number. The renumber function maps the provided input vertex IDs to 32-bit contiguous integers starting from 0. cuGraph still requires the renumbered vertex IDs to be representable in 32-bit integers. These limitations are being addressed and will be fixed soon.

cuGraph provides an auto-renumbering feature, enabled by default, during Graph creating.  Renumbered vertices are automatically un-renumbered.

cuGraph is constantly being updated and improved. Please see the [Transition Guide](TRANSITIONGUIDE.md) if errors are encountered with newer versions

## Graph Sizes and GPU Memory Size
The amount of memory required is dependent on the graph structure and the analytics being executed.  As a simple rule of thumb, the amount of GPU memory should be about twice the size of the data size.  That gives overhead for the CSV reader and other transform functions.  There are ways around the rule but using smaller data chunks.


|       Size        | Recommended GPU Memory |
|-------------------|------------------------|
| 500 million edges |  32GB                  |
| 250 million edges |  16 GB                 |




## Getting cuGraph
### Intro
There are 3 ways to get cuGraph :
1. [Quick start with Docker Demo Repo](#quick)
2. [Conda Installation](#conda)
3. [Build from Source](#source)


<a name="quick"></a>

## Quick Start
Please see the [Demo Docker Repository](https://hub.docker.com/r/rapidsai/rapidsai/), choosing a tag based on the NVIDIA CUDA version you’re running. This provides a ready to run Docker container with example notebooks and data, showcasing how you can utilize all of the RAPIDS libraries: cuDF, cuML, and cuGraph.


<a name="conda"></a>
### Conda
It is easy to install cuGraph using conda. You can get a minimal conda installation with [Miniconda](https://conda.io/miniconda.html) or get the full installation with [Anaconda](https://www.anaconda.com/download).

Install and update cuGraph using the conda command:

```bash

# CUDA 10.0
conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cugraph cudatoolkit=10.0

# CUDA 10.1
conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cugraph cudatoolkit=10.1

# CUDA 10.2
conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cugraph cudatoolkit=10.2
```

Note: This conda installation only applies to Linux and Python versions 3.6/3.7.


<a name="source"></a>
### Build from Source and Contributing

Please see our [guide for building cuGraph from source](SOURCEBUILD.md)</pr>

Please see our [guide for contributing to cuGraph](CONTRIBUTING.md).



## Documentation
Python API documentation can be generated from [docs](docs) directory.



------

## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aims to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="80%"/></p>

### Apache Arrow on GPU

The GPU version of [Apache Arrow](https://arrow.apache.org/) is a common API that enables efficient interchange of tabular data between processes running on the GPU. End-to-end computation on the GPU avoids unnecessary copying and converting of data off the GPU, reducing compute time and cost for high-performance analytics common in artificial intelligence workloads. As the name implies, cuDF uses the Apache Arrow columnar data format on the GPU. Currently, a subset of the features in Apache Arrow are supported.
