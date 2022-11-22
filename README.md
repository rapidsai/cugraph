<h1 align="center"; style="font-style: italic";>
  <br>
  <img src="img/cugraph_logo_2.png" alt="cuGraph" width="500">
</h1>

<div align="center">

`<a href="https://github.com/rapidsai/cugraph/blob/main/LICENSE">`
`<img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>`
`<img alt="GitHub tag (latest by date)" src="https://img.shields.io/github/v/tag/rapidsai/cugraph">`

<a href="https://github.com/rapidsai/cugraph/stargazers">
    <img src="https://img.shields.io/github/stars/rapidsai/cugraph"></a>
<img alt="Conda" src="https://img.shields.io/conda/dn/rapidsai/cugraph">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/rapidsai/cugraph">

<img alt="Conda" src="https://img.shields.io/conda/pn/rapidsai/cugraph" />

`<a href="https://rapids.ai/"><img src="img/rapids_logo.png" alt="RAPIDS" width="125">``</a>`

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

---

## Table of content

- Getting packages
  - [Getting cuGraph Packages](./readme_pages/getting_cugraph.md)
  - [Contributing to cuGraph](./readme_pages/CONTRIBUTING.md)
- General
  - [Latest News](./readme_pages/news.md)
  - [Current list of algorithms](./readme_pages/algorithms.md)
  - [BLOGs and Presentation](./docs/cugraph/source/basics/cugraph_blogs.rst)
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

`<br><br>`

---

<img src="img/Stack2.png" alt="Stack" width="800">

---

The [RAPIDS](https://rapids.ai) cuGraph library is a collection of GPU accelerated graph algorithms that process data found in [GPU DataFrames](https://github.com/rapidsai/cudf).  The vision of cuGraph is _to make graph analysis ubiquitous to the point that users just think in terms of analysis and not technologies or frameworks_.  To realize that vision, cuGraph operates, at the Python layer, on GPU DataFrames, thereby allowing for seamless passing of data between ETL tasks in [cuDF](https://github.com/rapidsai/cudf) and machine learning tasks in [cuML](https://github.com/rapidsai/cuml).  Data scientists familiar with Python will quickly pick up how cuGraph integrates with the Pandas-like API of cuDF.  Likewise, users familiar with NetworkX will quickly recognize the NetworkX-like API provided in cuGraph, with the goal to allow existing code to be ported with minimal effort into RAPIDS.

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

---
