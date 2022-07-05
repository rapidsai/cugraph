
# cuGraph Centrality Notebooks

![GraphAnalyticsFigure](../../img/zachary_graph_centrality.png)

cuGraph Centrality notebooks contain a collection of Jupyter Notebooks that demonstrate algorithms to identify and quantify importance of nodes to the structure of the graph.  In the diagram above, the highlighted nodes are highly important and are likely answers to questions like:

* Which nodes have the highest degree (most direct links) ?
* Which nodes are on the most efficient paths through the graph?
* Which nodes connect the most important nodes to each other?

But which nodes are most important? The answer depends on which measure/algorithm is run.  Manipulation of the data before or after the graph analytic is not covered here.   Extended, more problem focused, notebooks are being created and available https://github.com/rapidsai/notebooks-extended

## Summary

|Notebook(s)          |Algorithm                                                     |Description                                                  |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Centrality](centrality/Centrality.ipynb)                    |Degree Centrality| node with the most direct connections|
| [Centrality](centrality/Centrality.ipynb), [Betweenness](centrality/Betweenness.ipynb)                    |Betweenness Centrality|Number of shortest paths through the node|
|[Centrality](centrality/Centrality.ipynb)|Eigenvector Centrality| measure of connectivity to other important nodes (which also have high connectivity) often referred to as the influence measure of a node|
|[Centrality](centrality/Centrality.ipynb), [Katz](centrality/Katz.ipynb)                                         |Katz Centrality|Similar to Eigenvector but has tweaks to measure more weakly connected graph  |
|[Centrality](centrality/Centrality.ipynb), [Pagerank](../../link_analysis/Pagerank.ipynb)                                         |Pagerank |Classified as both a link analysis and centrality measure by quantifying incoming links from central nodes.  |

## RAPIDS notebooks

Visit the main RAPIDS [notebooks](https://github.com/rapidsai/notebooks) repo for a listing of all notebooks across all RAPIDS libraries.

## Requirements

Running the example in these notebooks requires:

* The latest version of RAPIDS with cuGraph.
  * Download via Docker, Conda (See [__Getting Started__](https://rapids.ai/start.html))
  
* cuGraph is dependent on the latest version of cuDF.  Please install all components of RAPIDS
* Python 3.8+
* A system with an NVIDIA GPU:  Pascal architecture or better
* CUDA 11.4+
* NVIDIA driver 450.51+

| Author Credit |    Date    |  Update          | cuGraph Version |  Test Hardware |
| --------------|------------|------------------|-----------------|----------------|
| Brad Rees     | 04/19/2021 | created          | 0.19            | GV100, CUDA 11.0
| Don Acosta    | 07/05/2022 | tested / updated | 22.08 nightly   | DGX Tesla V100 CUDA 11.5

## Copyright

Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");  you may not use this file except in compliance with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.

![RAPIDS](../../img/rapids_logo.png)
