# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuGraph - GPU Graph Analytics</div>

[![Build Status](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cugraph/job/branches/job/cugraph-branch-pipeline/badge/icon)](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cugraph/job/branches/job/cugraph-branch-pipeline/)

The [RAPIDS](https://rapids.ai) cuGraph library is a collection of GPU accelerated graph algorithms that process data found in GPU DataFrames - see [cuDF](https://github.com/rapidsai/cudf).  The vision of RAPIDS cuGraph is _to make graph analysis ubiquitous to the point that users just think in terms of analysis and not technologies or frameworks_.  To realize that vision, cuGraph operators, at the Python layer, on GPU DataFrames, allowing for seamless passing of data between ETL tasks in cuDF and machine learning tasks in cuML.  Data scientist familiar with Python will quickly pick up how cuGraph integrates with the Pandas-like API of cuDF.  For user familiar with NetworkX, cuGraph provides a NetworkX-like API.  The goal being to allow existing code to be ported with minimal effort into RAPIDS.  

For users familar with C/CUDA and graph structures, we also provide a C++ API.  There is less type and structure checking at the C layer.  

 For more project details, see [rapids.ai](https://rapids.ai/).

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/cudf/blob/master/README.md) ensure you are on the latest branch.



```markdown
import cugraph

# read data into a cuDF DataFrame using read_csv
gdf = cudf.read_csv("graph_data.csv", names=["src", "dst"], dtype=["int32", "int32"] )

# We now have data in a COO format (edge pairs)
# create a Graph using the source (src) and destination (dst) vertex pairs the GDF  
G = cugraph.Graph()
G.add_edge_list(gdf, source='src', destination='dst')

# Let's now get the PageRank score of each vertex by calling cugraph.pagerank
gdf_page = cugraph.pagerank(G)

# Let's look at the PageRank Score (only do this on small graphs)
for i in range(len(gdf_page)):
	print("vertex " + str(gdf_page['vertex'][i]) + 
		" PageRank is " + str(gdf_page['pagerank'][i]))  
```


## Supported Algorithms

| Category     | Algorithm                              | Sacle        |  Description                                                  |
| ------------ | -------------------------------------- | ------------ | ------------------------------------------------------------- |
| Centrality                                                                                                                        ||||
|              | Katz                                   | Single-GPU   | Compute the Katz centrality for every vertex                  |
|              | Betweenness Centrality                 | Single-GPU   | Compute the Betweenness Centrality of every vertex            |
| Community    |                                        |              |                                                               |
|              | Louvain                                | Single-GPU   | Identify clusters in a graph using the Louvain algorithm      |
|              | Ensemble Clustering for Graphs         | Single-GPU   | An Ensemble variation of Louvain                              |
|              | Spectral-Clustering - Balanced Cut     | Single-GPU   | Identify clusters using Spectral Clustering Balanced Cut      |
|              | Spectral-Clustering                    | Single-GPU   | Identify clusters using Spectral Clustering Modularity Modularity |
|              | Subgraph Extraction                    | Single-GPU   | Induce a subgraph that includes only the specified vertices   |
|              | Triangle Counting                      | Single-GPU   | Count the number of Triangle in a graph                       |
| Components   |                                        |              |                                                               |
|              | Weakly Connected Components            | Single-GPU   | Find weakly connected components in a graph                   |
|              | Strongly Connected Components          | Single-GPU   | Find strongly connected components in a graph                 |
| Core         |                                        |              |                                                               |
|              | K-Core                                 | Single-GPU   | Identify the K-Core clusters in a graph                       |
|              | Core Number                            | Single-GPU   | Compute the max K core number                                 |
|              | K-Truss                                | Single-GPU   | Identify clusters in a graph using the K-Truss algorithm      |
| Link Analysis|                                        |              |                                                               |
|              | Pagerank                               | Single-GPU   | Compute the PageRank score of every vertex in a graph         |
|              | Personal Pagerank                      | Single-GPU   | Compute the Personal PageRank of every vertex in a graph      |
| Link Prediction |                                     |              |                                                               |
|              | Jacard Similarity                      | Single-GPU   | Compute vertex similarity score using Jaccard Similarity      |
|              | Weighted Jacard Similarity             | Single-GPU   | Compute vertex similarity score using Weighted Jaccard Similarity |
|              | Overlap Similarity                     | Single-GPU   | Compute vertex similarity score using the Overlap Coefficient |
| Traversal    |                                        |              |                                                               |
|              | Breadth First Search (BFS)             | Single-GPU   | Compute the BFS path from a starting vertex to every other vertex in a graph |
|              | Single Source Shortest Path (SSSP)     | Single-GPU   | Compute the shortest path from a starting vertex to every other vertex |
| Structure    |                                        |              |                                                               |
|              | Renumbering                            | Single-GPU   | Renumber the vertex IDs, a vertex can be one or more columns  |
|              | Symmetrize                             | Single-GPU   | Symmetrize the edges in a graph                               |

## Supported Graph
| Type            |  Description                                        |
| --------------- | --------------------------------------------------- |
| Graph           | An undirected Graph                                 |
| DiGraph         | A Directed Graph                                    |


## cuGraph Notice
The current version of cuGraph has some limitations:

- Vertex IDs need to be 32-bit integers.
- Vertex IDs are expected to be contiguous integers starting from 0.
--  If the starting index is not zero, cuGraph will add disconnected vertices to fill in the missing range

cuGraph provides the renumber function to mitigate this problem. Input vertex IDs for the renumber function can be any type, can be non-contiguous, and can start from an arbitrary number. The renumber function maps the provided input vertex IDs to 32-bit contiguous integers starting from 0. cuGraph still requires the renumbered vertex IDs to be representable in 32-bit integers. These limitations are being addressed and will be fixed soon.

cuGraph prvides an auto-renumbering feature, enabled by default, during Graph creating.  Renumbered vertices are automaticaly un-renumbered.

cuGraph is constantly being updatred and improved. Please see the [Transition Guide](TRANSITIONGUIDE.md) is errors are encountered with newer versions



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

Please see our [guide for building and contributing to cuGraph](CONTRIBUTING.md).



## Documentation
Python API documentation can be generated from [docs](docs) directory.



------

## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="80%"/></p>

### Apache Arrow on GPU

The GPU version of [Apache Arrow](https://arrow.apache.org/) is a common API that enables efficient interchange of tabular data between processes running on the GPU. End-to-end computation on the GPU avoids unnecessary copying and converting of data off the GPU, reducing compute time and cost for high-performance analytics common in artificial intelligence workloads. As the name implies, cuDF uses the Apache Arrow columnar data format on the GPU. Currently, a subset of the features in Apache Arrow are supported.
