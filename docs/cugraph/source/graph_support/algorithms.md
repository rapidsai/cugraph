# List of Supported and Planned Algorithms

## Supported Graph

| Type       | Description                                                 |
| ---------- | ----------------------------------------------------------- |
| Graph      | A directed or undirected Graph (use directed={True, False}) |
| Multigraph | A Graph with multiple edges between a vertex pair           |
|            |                                                             |

ALL Algorithms support Graphs and MultiGraph (directed and undirected)

---

<br>

# Supported Algorithms

_Italic_ algorithms are planned for future releases.

Note: Multi-GPU, or MG, includes support for Multi-Node Multi-GPU (also called MNMG).

| Category          | Notebooks                          | Scale               | Notes                                                           |
| ----------------- | ---------------------------------- | ------------------- | --------------------------------------------------------------- |
| [Centrality](./algorithms/Centrality.md)        | [Centrality](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/centrality/Centrality.ipynb)                                   |                     |                                                                 |
|                   | [Katz](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/centrality/Katz.ipynb)                               | __Multi-GPU__ |                                                                 |
|                   | [Betweenness Centrality](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/centrality/Betweenness.ipynb)             | __Multi-GPU__          | MG as of 23.06                                            |
|                   | [Edge Betweenness Centrality](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/centrality/Betweenness.ipynb)        | Single-GPU          | MG planned for 23.08                                            |
|                   | [Eigenvector Centrality](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/centrality/Eigenvector.ipynb)             | __Multi-GPU__ |                                                                 |
|                   | [Degree Centrality](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/centrality/Degree.ipynb)                  | __Multi-GPU__ | Python only                                                     |
| Community         |                                    |                     |                                                                 |
|                   | [Leiden](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/community/Louvain.ipynb)                             | __Multi-GPU__          | MG as of 23.06                                            |
|                   | [Louvain](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/community/Louvain.ipynb)                            | __Multi-GPU__ |                                                                 |
|                   | [Ensemble Clustering for Graphs](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/community/ECG.ipynb)     | Single-GPU          |                                                                 |
|                   | [Spectral-Clustering - Balanced Cut](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/community/Spectral-Clustering.ipynb) | Single-GPU          |                                                                 |
|                   | [Spectral-Clustering - Modularity](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/community/Spectral-Clustering.ipynb)   | Single-GPU          |                                                                 |
|                   | [Subgraph Extraction](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/community/Subgraph-Extraction.ipyn)                | Single-GPU          |                                                                 |
|                   | [Triangle Counting](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/community/Triangle-Counting.ipynb)                  | __Multi-GPU__ |                                                                 |
|                   | [K-Truss](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/community/ktruss.ipynb)                            | Single-GPU          |                                                                 |
| Components        |                                    |                     |                                                                 |
|                   | [Weakly Connected Components](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/components/ConnectedComponents.ipynb)        | __Multi-GPU__ |                                                                 |
|                   | [Strongly Connected Components](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/components/ConnectedComponents.ipynb)      | Single-GPU          |                                                                 |
| Core              |                                    |                     |                                                                 |
|                   | [K-Core](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/cores/kcore.ipynb)                             | **Multi-GPU** |                                                                 |
|                   | [Core Number](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/cores/core-number.ipynb)                        | **Multi-GPU** |                                                                 |
| _Flow_          |                                    |                     |                                                                 |
|                   | _MaxFlow_                        | ---                 |                                                                 |
| _Influence_     |                                    |                     |                                                                 |
|                   | _Influence Maximization_         | ---                 |                                                                 |
| Layout            |                                    |                     |                                                                 |
|                   | [Force Atlas 2](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/layout/Force-Atlas2.ipynb)                      | Single-GPU          |                                                                 |
| Linear Assignment |                                    |                     |                                                                 |
|                   | [Hungarian]()                          | Single-GPU          | [README](cpp/src/linear_assignment/README-hungarian.md)            |
| Link Analysis     |                                    |                     |                                                                 |
|                   | [Pagerank](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/link_analysis/Pagerank.ipynb)                           | __Multi-GPU__ | [C++ README](cpp/src/centrality/README.md#Pagerank)                |
|                   | [Personal Pagerank]()                  | __Multi-GPU__ | [C++ README](cpp/src/centrality/README.md#Personalized-Pagerank)   |
|                   | [HITS](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/link_analysis/HITS.ipynb)                               | __Multi-GPU__ |                                                                 |
| [Link Prediction](./algorithms/Similarity.md)   |                                    |                     |                                                                 |
|                   | [Jaccard Similarity](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/link_prediction/Jaccard-Similarity.ipynb)                 | __Multi-GPU__      | Directed graph only                         |
|                   | [Weighted Jaccard Similarity](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/link_prediction/Jaccard-Similarity.ipynb)        | Single-GPU          |                                                                 |
|                   | [Overlap Similarity](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/link_prediction/Overlap-Similarity.ipynb)                 | **Multi-GPU** |                                                   |
|                   | Sorensen Coefficient - Coming Soon| **Multi-GPU** |                                                   |
|                   | _Local Clustering Coefficient_   | ---                 |                                                                 |
| Sampling          |                                    |                     |                                                                 |
|                   | [Uniform Random Walks RW](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/sampling/RandomWalk.ipynb)          | __Multi-GPU__ |                                                                 |
|                   | *Biased Random Walks (RW)*       | ---                 |                                                                 |
|                   | Egonet                             | __Multi-GPU__ |                                                                 |
|                   | Node2Vec                           | Single-GPU          |                                             |
|                   | Uniform Neighborhood sampling      | __Multi-GPU__ |                                                                 |
| Traversal         |                                    |                     |                                                                 |
|                   | Breadth First Search (BFS)         | __Multi-GPU__ | with cutoff support``[C++ README](cpp/src/traversal/README.md#BFS) |
|                   | Single Source Shortest Path (SSSP) | __Multi-GPU__ | [C++ README](cpp/src/traversal/README.md#SSSP)                     |
|                   | _ASSP / APSP_                    | ---                 |                                                                 |
| Tree              |                                    |                     |                                                                 |
|                   | Minimum Spanning Tree              | Single-GPU          |                                                                 |
|                   | Maximum Spanning Tree              | Single-GPU          |                                                                 |
| Other             |                                    |                     |                                                                 |
|                   | Renumbering                        | __Multi-GPU__ | multiple columns, any data type                                 |
|                   | Symmetrize                         | __Multi-GPU__ |                                                                 |
|                   | Path Extraction                    |                     | Extract paths from BFS/SSP results in parallel                  |
|                   | Two Hop Neighbors                  | __Multi-GPU__ |                                                                 |
| Data Generator    |                                    |                     |                                                                 |
|                   | RMAT                               | __Multi-GPU__ |                                                                 |
|                   | _Barabasi-Albert_                | ---                 |                                                                 |
|                   |                                    |                     |                                                                 |
