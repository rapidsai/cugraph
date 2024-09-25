# Supported Algorithms


The nx-cugraph backend to NetworkX connects
[pylibcugraph](../../readme_pages/pylibcugraph.md) (cuGraph's low-level python
interface to its CUDA-based graph analytics library) and
[CuPy](https://cupy.dev/) (a GPU-accelerated array library) to NetworkX's
familiar and easy-to-use API.

Below is the list of algorithms that are currently supported in nx-cugraph.

### [Algorithms](https://networkx.org/documentation/latest/reference/algorithms/index.html)


| **Centrality**    | **Centrality**   |
|------------------------------|------------------------------|
| betweenness_centrality       |  betweenness_centrality      |
| edge_betweenness_centrality  | in_degree_centrality         |
| degree_centrality            |  edge_betweenness_centrality |
| in_degree_centrality         | in_degree_centrality         |
| out_degree_centrality        |  in_degree_centrality        |
| eigenvector_centrality       |  in_degree_centrality        |
| katz_centrality              |  in_degree_centrality        |


To request nx-cugraph backend support for a NetworkX API that is not listed
above, visit the [cuGraph GitHub repo](https://github.com/rapidsai/cugraph).
