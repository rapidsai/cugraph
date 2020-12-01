# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuGraph - GPU Graph Analytics</div>

[![Build Status](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cugraph/job/branches/job/cugraph-branch-pipeline/badge/icon)](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cugraph/job/branches/job/cugraph-branch-pipeline/)

The [RAPIDS](https://rapids.ai) cuGraph library is a collection of GPU accelerated graph algorithms that process data found in [GPU DataFrames](https://github.com/rapidsai/cudf).  The vision of cuGraph is _to make graph analysis ubiquitous to the point that users just think in terms of analysis and not technologies or frameworks_.  To realize that vision, cuGraph operates, at the Python layer, on GPU DataFrames, thereby allowing for seamless passing of data between ETL tasks in [cuDF](https://github.com/rapidsai/cudf) and machine learning tasks in [cuML](https://github.com/rapidsai/cuml).  Data scientists familiar with Python will quickly pick up how cuGraph integrates with the Pandas-like API of cuDF.  Likewise, users familiar with NetworkX will quickly recognize the NetworkX-like API provided in cuGraph, with the goal to allow existing code to be ported with minimal effort into RAPIDS.  For users familiar with C++/CUDA and graph structures, a C++ API is also provided.  However, there is less type and structure checking at the C++ layer.

 For more project details, see [rapids.ai](https://rapids.ai/).

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/cugraph/blob/main/README.md) ensure you are on the latest branch.



```python
import cugraph

# read data into a cuDF DataFrame using read_csv
gdf = cudf.read_csv("graph_data.csv", names=["src", "dst"], dtype=["int32", "int32"])

# We now have data as edge pairs
# create a Graph using the source (src) and destination (dst) vertex pairs
G = cugraph.Graph()
G.from_cudf_edgelist(gdf, source='src', destination='dst')

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
|              | Katz                                   | Multi-GPU    |                     |
|              | Betweenness Centrality                 | Single-GPU   |                     |
|              | Edge Betweenness Centrality            | Single-GPU   |                     |
| Community    |                                        |              |                     |
|              | Leiden                                 | Single-GPU   |                     |
|              | Louvain                                | Multi-GPU    |                     |
|              | Ensemble Clustering for Graphs         | Single-GPU   |                     |
|              | Spectral-Clustering - Balanced Cut     | Single-GPU   |                     |
|              | Spectral-Clustering - Modularity       | Single-GPU   |                     |
|              | Subgraph Extraction                    | Single-GPU   |                     |
|              | Triangle Counting                      | Single-GPU   |                     |
|              | K-Truss                                | Single-GPU   |                     |
| Components   |                                        |              |                     |
|              | Weakly Connected Components            | Single-GPU   |                     |
|              | Strongly Connected Components          | Single-GPU   |                     |
| Core         |                                        |              |                     |
|              | K-Core                                 | Single-GPU   |                     |
|              | Core Number                            | Single-GPU   |                     |
| Layout       |                                        |              |                     |
|              | Force Atlas 2                          | Single-GPU   |                     |
| Link Analysis|                                        |              |                     |
|              | Pagerank                               | Multi-GPU    |                     |
|              | Personal Pagerank                      | Multi-GPU    |                     |
|              | HITS                      				| Single-GPU   | leverages Gunrock   |
| Link Prediction |                                     |              |                     |
|              | Jaccard Similarity                     | Single-GPU   |                     |
|              | Weighted Jaccard Similarity            | Single-GPU   |                     |
|              | Overlap Similarity                     | Single-GPU   |                     |
| Traversal    |                                        |              |                     |
|              | Breadth First Search (BFS)             | Multi-GPU    |                     |
|              | Single Source Shortest Path (SSSP)     | Multi-GPU    |                     |
| Structure    |                                        |              |                     |
|              | Renumbering                            | Single-GPU   | multiple columns, any data type  |
|              | Symmetrize                             | Multi-GPU    |                     |
| Other        |                                        |              |                     |
|              | Hungarian Algorithm                    | Single-GPU   |                     |
|              | Minimum Spanning Tree                  | Single-GPU   |                     |
|              | Maximum Spanning Tree                  | Single-GPU   |                     |

|  |  |

</br></br>
## Supported Graph
| Type            |  Description                                        |
| --------------- | --------------------------------------------------- |
| Graph           | An undirected Graph                                 |
| DiGraph         | A Directed Graph                                    |
| _Multigraph_      | _coming in 0.18_                                      |
| _MultiDigraph_    | _coming in 0.18_                                      |
|  |  |

</br></br>
## Supported Data Types
cuGraph supports the creation of a graph several data types:
* cuDF DataFrame
* Pandas DataFrame

cuGraph supports execution of graph algorithms from different graph objects
* cuGraph Graph classes
* NetworkX graph classes
* CuPy sparse matrix
* SciPy sparse matrix

cuGraph tries to match the return type based on the input type.  So a NetworkX input will return the same data type that NetworkX would have.


## cuGraph Notice
The current version of cuGraph has some limitations:

- Vertex IDs are expected to be contiguous integers starting from 0.

cuGraph provides the renumber function to mitigate this problem, which is by default automatically called when data is addted to a graph.  Input vertex IDs for the renumber function can be any type, can be non-contiguous, can be multiple columns, and can start from an arbitrary number. The renumber function maps the provided input vertex IDs to 32-bit contiguous integers starting from 0. cuGraph still requires the renumbered vertex IDs to be representable in 32-bit integers. These limitations are being addressed and will be fixed soon.

Additionally, when using the auto-renumbering feature, vertices are automatically un-renumbered in results.

cuGraph is constantly being updated and improved. Please see the [Transition Guide](TRANSITIONGUIDE.md) if errors are encountered with newer versions

## Graph Sizes and GPU Memory Size
The amount of memory required is dependent on the graph structure and the analytics being executed.  As a simple rule of thumb, the amount of GPU memory should be about twice the size of the data size.  That gives overhead for the CSV reader and other transform functions.  There are ways around the rule but using smaller data chunks.

|       Size        | Recommended GPU Memory |
|-------------------|------------------------|
| 500 million edges |  32 GB                 |
| 250 million edges |  16 GB                 |

The use of managed memory for oversubscription can also be used to exceed the above memory limitations.  See the recent blog on _Tackling Large Graphs with RAPIDS cuGraph and CUDA Unified Memory on GPUs_:  https://medium.com/rapids-ai/tackling-large-graphs-with-rapids-cugraph-and-unified-virtual-memory-b5b69a065d4


## Getting cuGraph
### Intro
There are 3 ways to get cuGraph :
1. [Quick start with Docker Demo Repo](#quick)
2. [Conda Installation](#conda)
3. [Build from Source](#source)




## Quick Start <a name="quick"></a>
Please see the [Demo Docker Repository](https://hub.docker.com/r/rapidsai/rapidsai/), choosing a tag based on the NVIDIA CUDA version you’re running. This provides a ready to run Docker container with example notebooks and data, showcasing how you can utilize all of the RAPIDS libraries: cuDF, cuML, and cuGraph.


### Conda <a name="conda"></a>
It is easy to install cuGraph using conda. You can get a minimal conda installation with [Miniconda](https://conda.io/miniconda.html) or get the full installation with [Anaconda](https://www.anaconda.com/download).

Install and update cuGraph using the conda command:

```bash

# CUDA 10.1
conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cugraph cudatoolkit=10.1

# CUDA 10.2
conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cugraph cudatoolkit=10.2

# CUDA 11.0
conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cugraph cudatoolkit=11.0
```

Note: This conda installation only applies to Linux and Python versions 3.7/3.8.


### Build from Source and Contributing <a name="source"></a>

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
