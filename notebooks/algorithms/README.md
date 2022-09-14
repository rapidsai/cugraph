# cuGraph Algorithm Notebooks

As all the algorithm Notebooks are updated and migrated to this area, they will show in this Readme. Until then they are available [here](../README.md)

![GraphAnalyticsFigure](../img/GraphAnalyticsFigure.jpg)

This repository contains a collection of Jupyter Notebooks that outline how to run various cuGraph analytics.   The notebooks do not address a complete data science problem.  The notebooks are simply examples of how to run the graph analytics.  Manipulation of the data before or after the graph analytic is not covered here.   Extended, more problem focused, notebooks are being created and available https://github.com/rapidsai/notebooks-extended

## Summary

| Folder          | Notebook                                                     | Description                                                  |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Centrality](centrality/README.md)      |                                                              |                                                              |
|                 | [Centrality](centrality/Centrality.ipynb)                    | Compute and compare multiple (currently 5) centrality scores |
|                 | [Katz](centrality/Katz.ipynb)                                | Compute the Katz centrality for every vertex                 |
|                 | [Betweenness](centrality/Betweenness.ipynb)                  | Compute both Edge and Vertex Betweenness centrality          |
|                 | [Degree](centrality/Degree.ipynb)                            | Compute Degree Centraility for each vertex                   |
|                 | [Eigenvector](centrality/Eigenvector.ipynb)                  | Compute Eigenvector for every vertex                         |
|[Community](community/README.md)       |                                                              |                                                              |
|                 | [Louvain](community/Louvain.ipynb)                           | Identify clusters in a graph using both the Louvain and Leiden algorithms     |
|                 | [ECG](community/ECG.ipynb)                                   | Identify clusters in a graph using the Ensemble Clustering for Graph |
|                 | [K-Truss](community/ktruss.ipynb)                                | Extracts the K-Truss cluster                                 |
|                 | [Spectral-Clustering](community/Spectral-Clustering.ipynb)   | Identify clusters in a  graph using Spectral Clustering with both<br> - Balanced Cut<br> - Modularity Modularity |
|                 | [Subgraph Extraction](community/Subgraph-Extraction.ipynb)   | Compute a subgraph of the existing graph including only the specified vertices |
|                 | [Triangle Counting](community/Triangle-Counting.ipynb)       | Count the number of Triangle in a graph                      |
|[Components](components/README.md)      |                                                              |                                                              |
|                 | [Connected Components](components/ConnectedComponents.ipynb) | Find weakly and strongly connected components in a graph     |
| [Cores](cores/README.md)            |                                                              |                                                              |
|                | [core-number](cores/Core-number.ipynb)   | Computes the core number for every vertex of a graph G. The core number of a vertex is a maximal subgraph that contains only that vertex and others of degree k or more. |
|                | [kcore](cores/kcore.ipynb)               |Find the k-core of a graph which is a maximal subgraph that contains nodes of degree k or more.|
Layout            |                                                              |                                                              |
|                | [Force-Atlas2](layout/Force-Atlas2.ipynb)   |A large graph visualization achieved with cuGraph. |
| [Link Analysis](link_analysis/README.md)   |                                                              |                                                              |
|                 | [Pagerank](link_analysis/Pagerank.ipynb)                     | Compute the PageRank of every vertex in a graph              |
|                 | [HITS](link_analysis/HITS.ipynb)                             | Compute the HITS' Hub and Authority scores for every vertex in a graph              |
| [Link Prediction](link_prediction/README.md) |                                                              |                                                              |
|                 | [Jaccard Similarity](algorithms/link_prediction/Jaccard-Similarity.ipynb) | Compute vertex similarity score using both:<br />- Jaccard Similarity<br />- Weighted Jaccard |
|                 | [Overlap Similarity](algorithms/link_prediction/Overlap-Similarity.ipynb) | Compute vertex similarity score using the Overlap Coefficient |
| [Sampling](sampling/README.md)        |
|                 | [Random Walk](sampling/RandomWalk.ipynb)                     | Compute Random Walk for a various number of seeds and path lengths |
| [Traversal](traversal/README.md)       |                                                              |                                                              |
|                 | [BFS](traversal/BFS.ipynb)                                   | Compute the Breadth First Search path from a starting vertex to every other vertex in a graph |
|                 | [SSSP](traversal/SSSP.ipynb)                                 | Single Source Shortest Path  - compute the shortest path from a starting vertex to every other vertex |
| [Structure](structure/README.md)       |                                                              |                                                              |
|                 | [Renumbering](structure/Renumber.ipynb) <br> [Renumbering 2](structure/Renumber-2.ipynb) | Renumber the vertex IDs in a graph (two sample notebooks)    |
|                 | [Symmetrize](structure/Symmetrize.ipynb)                     | Symmetrize the edges in a graph                              |

[System Requirements](../README.md#requirements)

| Author Credit |    Date    |  Update          | cuGraph Version |  Test Hardware |
| --------------|------------|------------------|-----------------|----------------|
| Brad Rees     | 04/19/2021 | created          | 0.19            | GV100, CUDA 11.0
| Don Acosta    | 08/02/2022 | tested / updated | 22.08 nightly   | DGX Tesla V100 CUDA 11.5

### Copyright

Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");  you may not use this file except in compliance with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.

![RAPIDS](../img/rapids_logo.png)
