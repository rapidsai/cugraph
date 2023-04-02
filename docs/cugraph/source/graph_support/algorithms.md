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

| Category          | Algorithm                          | Scale               | Notes                                                           |
| ----------------- | ---------------------------------- | ------------------- | --------------------------------------------------------------- |
| Centrality        |                                    |                     |                                                                 |
|                   | Katz                               | __Multi-GPU__ |                                                                 |
|                   | Betweenness Centrality             | Single-GPU          | MG planned for 23.02                                            |
|                   | Edge Betweenness Centrality        | Single-GPU          | MG planned for 23.02                                            |
|                   | Eigenvector Centrality             | __Multi-GPU__ |                                                                 |
|                   | Degree Centrality                  | __Multi-GPU__ | Python only                                                     |
| Community         |                                    |                     |                                                                 |
|                   | Leiden                             | Single-GPU          | MG planned for 23.02                                            |
|                   | Louvain                            | __Multi-GPU__ |                                                                 |
|                   | Ensemble Clustering for Graphs     | Single-GPU          |                                                                 |
|                   | Spectral-Clustering - Balanced Cut | Single-GPU          |                                                                 |
|                   | Spectral-Clustering - Modularity   | Single-GPU          |                                                                 |
|                   | Subgraph Extraction                | Single-GPU          |                                                                 |
|                   | Triangle Counting                  | __Multi-GPU__ |                                                                 |
|                   | K-Truss                            | Single-GPU          |                                                                 |
| Components        |                                    |                     |                                                                 |
|                   | Weakly Connected Components        | __Multi-GPU__ |                                                                 |
|                   | Strongly Connected Components      | Single-GPU          |                                                                 |
| Core              |                                    |                     |                                                                 |
|                   | K-Core                             | **Multi-GPU** |                                                                 |
|                   | Core Number                        | **Multi-GPU** |                                                                 |
| _Flow_          |                                    |                     |                                                                 |
|                   | _MaxFlow_                        | ---                 |                                                                 |
| _Influence_     |                                    |                     |                                                                 |
|                   | _Influence Maximization_         | ---                 |                                                                 |
| Layout            |                                    |                     |                                                                 |
|                   | Force Atlas 2                      | Single-GPU          |                                                                 |
| Linear Assignment |                                    |                     |                                                                 |
|                   | Hungarian                          | Single-GPU          | [README](cpp/src/linear_assignment/README-hungarian.md)            |
| Link Analysis     |                                    |                     |                                                                 |
|                   | Pagerank                           | __Multi-GPU__ | [C++ README](cpp/src/centrality/README.md#Pagerank)                |
|                   | Personal Pagerank                  | __Multi-GPU__ | [C++ README](cpp/src/centrality/README.md#Personalized-Pagerank)   |
|                   | HITS                               | __Multi-GPU__ |                                                                 |
| Link Prediction   |                                    |                     |                                                                 |
|                   | Jaccard Similarity                 | **Multi-GPU**      | MG as of 22.12<br />Directed graph only                         |
|                   | Weighted Jaccard Similarity        | Single-GPU          |                                                                 |
|                   | Overlap Similarity                 | **Multi-GPU** | MG as of 22.12                                                  |
|                   | Sorensen Coefficient               | **Multi-GPU** | MG as of 22.12                                                  |
|                   | _Local Clustering Coefficient_   | ---                 |                                                                 |
| Sampling          |                                    |                     |                                                                 |
|                   | Uniform Random Walks (RW)          | **Multi-GPU** |                                                                 |
|                   | *Biased Random Walks (RW)*       | ---                 |                                                                 |
|                   | Egonet                             | **Multi-GPU** |                                                                 |
|                   | Node2Vec                           | Single-GPU          | MG planned for 23.02                                            |
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
