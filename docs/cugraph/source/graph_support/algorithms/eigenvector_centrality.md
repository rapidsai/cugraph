# Eigenvector Centrality

Eigenvector centrality computes the centrality for a vertex based on the
centrality of its neighbors. The Eigenvector of a node measures influence within a graph by taking into account a vertex's connections to other highly connected vertices.


See [Eigenvector Centrality on Wikipedia](https://en.wikipedia.org/wiki/Eigenvector_centrality) for more details on the algorithm.

The eigenvector centrality for node i is the
i-th element of the vector x defined by the eigenvector equation.

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/51c506ef0f23db1086b00ff6e5da847ff53cf5e9" />


Where M(v) is the adjacency list for the set of vertices(v) and λ is a constant.

[Learn more about EigenVector Centrality](https://www.sci.unich.it/~francesc/teaching/network/eigenvector.html)

## When to use Eigenvector Centrality
* When the quality and quantity of edges matters, in other words, connections to other high-degree nodes is important
* To calculate influence in nuanced networks like social and financial networks. 

## When not to use Eigenvector Centrality
* in graphs with many disconnected groups
* in graphs containing many distinct and different communities 
* in networks with negative weights
* in huge networks eigenvector centrality can become computationally infeasible in single threaded systems.


## How computationally expensive is it?
While cuGraph's parallelism migigates run time, [Big O notation](https://en.wikipedia.org/wiki/Big_O_notation) is still the standard to compare algorithm costs.

O(VE) where V is the number of vertices(nodes) and Eis the number of edges.

___
Copyright (c) 2023, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");  you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
___

