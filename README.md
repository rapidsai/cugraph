<h1 align="center"; style="font-style: italic";>
  <br>
  <img src="img/cugraph_logo_2.png" alt="cuGraph" width="500">
</h1>

<div align="center">

[![Build Status](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cugraph/job/branches/job/cugraph-branch-pipeline/badge/icon)](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cugraph/job/branches/job/cugraph-branch-pipeline/)
<a href="https://github.com/rapidsai/cugraph/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
<img alt="GitHub tag (latest by date)" src="https://img.shields.io/github/v/tag/rapidsai/cugraph">

<a href="https://github.com/rapidsai/cugraph/stargazers">
    <img src="https://img.shields.io/github/stars/rapidsai/cugraph"></a>
<img alt="Conda" src="https://img.shields.io/conda/dn/rapidsai/cugraph">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/rapidsai/cugraph">

<img alt="Conda" src="https://img.shields.io/conda/pn/rapidsai/cugraph" />

<a href="https://rapids.ai/"><img src="img/rapids_logo.png" alt="RAPIDS" width="125"></a>

</div>

<br>

[RAPIDS](https://rapids.ai) cuGraph is a monorepo that represents a collection of packages focused on GPU-accelerated graph analytics, including support for property graphs, remote (graph as a service) operations, and graph neural networks (GNNs).  cuGraph supports the creation and manipulation of graphs followed by the execution of scalable fast graph algorithms.

<div align="center">

[Getting cuGraph](./readme_pages/getting_cugraph.md) *
[Graph Algorithms](./readme_pages/algorithms.md) *
[Graph Service](./readme_pages/cugraph_service.md) *
[Property Graph](./readme_pages/property_graph.md) *
[GNN Support](./readme_pages/gnn_support.md)

</div>

-----

## Table of content
- Getting packages
  - [Getting cuGraph Packages](./readme_pages/getting_cugraph.md)
  - [Contributing to cuGraph](./readme_pages/CONTRIBUTING.md)
- General
  - [Latest News](./readme_pages/news.md)
  - [Current list of algorithms](./readme_pages/algorithms.md)
  - [Blogs and Presentation](./docs/cugraph/source/basics/cugraph_blogs.rst)
  - [Performance](./readme_pages/performance/performance.md)
- Packages
  - [cuGraph Python](./readme_pages/cugraph_python.md)
    - [Property Graph](./readme_pages/property_graph.md)
    - [External Data Types](./readme_pages/data_types.md)
  - [pylibcugraph](./readme_pages/pylibcugraph.md)
  - [libcugraph (C/C++/CUDA)](./readme_pages/libcugraph.md)
  - [cugraph-service](./readme_pages/cugraph_service.md)
  - [cugraph-dgl](./readme_pages/cugraph_dgl.md)
  - [cugraph-ops](./readme_pages/cugraph_ops.md)
- API Docs
  - Python
    - [Python Nightly](https://docs.rapids.ai/api/cugraph/nightly/)
    - [Python Stable](https://docs.rapids.ai/api/cugraph/stable/)
  - C++
    - [C++ Nightly](https://docs.rapids.ai/api/libcugraph/nightly/)
    - [C++ Stable](https://docs.rapids.ai/api/libcugraph/stable/)
- References
  - [RAPIDS](https://rapids.ai/)
  - [ARROW](https://arrow.apache.org/)
  - [DASK](https://www.dask.org/)

<br><br>

-----

<img src="img/Stack2.png" alt="Stack" width="800">

[RAPIDS](https://rapids.ai) cuGraph is a collection of GPU-accelerated graph algorithms and services. At the Python layer, cuGraph operates on [GPU DataFrames](https://github.com/rapidsai/cudf), thereby allowing for seamless passing of data between ETL tasks in [cuDF](https://github.com/rapidsai/cudf) and machine learning tasks in [cuML](https://github.com/rapidsai/cuml). Data scientists familiar with Python will quickly pick up how cuGraph integrates with the Pandas-like API of cuDF.  Likewise, users familiar with NetworkX will quickly recognize the NetworkX-like API provided in cuGraph, with the goal to allow existing code to be ported with minimal effort into RAPIDS. To similfy integration, cuGraph also support data found in [Pandas DataFrame](https://pandas.pydata.org/), [NetworkX Graph Objects](https://networkx.org/) and several other formats.

While the high-level cugraph python API provides an easy-to-use and familiar interface for data scientists that's consistent with other RAPIDS libraries in their workflow, some use cases require access to lower-level graph theory concepts.  For these users, we provide an additional Python API called pylibcugraph, intended for applications that require a tighter integration with cuGraph at the Python layer with fewer dependencies.  Users familiar with C/C++/CUDA and graph structures can access libcugraph and libcugraph_c for low level integration outside of python.

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/cugraph/blob/main/README.md) ensure you are on the latest branch.



As an example, the following Python snippet loads graph data and computes PageRank:

```python
import cudf
import cugraph

# read data into a cuDF DataFrame using read_csv
gdf = cudf.read_csv("graph_data.csv", names=["src", "dst"], dtype=["int32", "int32"])

# We now have data as edge pairs
# create a Graph using the source (src) and destination (dst) vertex pairs
G = cugraph.Graph()
G.from_cudf_edgelist(gdf, source='src', destination='dst')

# Let's now get the PageRank score of each vertex by calling cugraph.pagerank
df_page = cugraph.pagerank(G)

# Let's look at the top 10 PageRank Score
df_page.sort_values('pagerank', ascending=False).head(10)

```

</br>

[Why cuGraph does not support Method Cascading]()



------
# Projects that use cuGraph

(alphabetical order)
* ArangoDB - a free and open-source native multi-model database system  - https://www.arangodb.com/
* CuPy - "NumPy/SciPy-compatible Array Library for GPU-accelerated Computing with Python" -  https://cupy.dev/
* Memgraph - In-memory Graph database - https://memgraph.com/
* ScanPy - a scalable toolkit for analyzing single-cell gene expression data - https://scanpy.readthedocs.io/en/stable/

(please post an issue if you have a project to add to this list)



------
<br>

## <div align="center"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science <a name="rapids"></a>


The RAPIDS suite of open source software libraries aims to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="50%"/></p>

For more project details, see [rapids.ai](https://rapids.ai/).

<br><br>
### Apache Arrow on GPU  <a name="arrow"></a>

The GPU version of [Apache Arrow](https://arrow.apache.org/) is a common API that enables efficient interchange of tabular data between processes running on the GPU. End-to-end computation on the GPU avoids unnecessary copying and converting of data off the GPU, reducing compute time and cost for high-performance analytics common in artificial intelligence workloads. As the name implies, cuDF uses the Apache Arrow columnar data format on the GPU. Currently, a subset of the features in Apache Arrow are supported.