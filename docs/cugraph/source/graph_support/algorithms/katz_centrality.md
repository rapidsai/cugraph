# Katz Centrality

Katz centrality is a measure of the relative importance of a vertex within the graph based on measuring the influence across the total number of walks between vertex pairs. Katz is similar to Eigenvector centrality. The main difference is that Katz also takes into account indirect relationships. The Katz calculation includes a user-controlled attenuation variable that controls the weight of indirect relationships. Otherwise it shares many of the advantages and disadvantages of Eigenvector centrality.

$C_{katz}(i) = \sum_{k=1}^{\infty} \sum_{j=1}^{n} \alpha ^k(A^k)_{ji}$

See [Katz on Wikipedia](https://en.wikipedia.org/wiki/Katz_centrality) for more details on the algorithm.

## When to use Katz Centrality
* in disconnected graphs
* in sparse graphs
* in graphs with multi-hop propogation like innovation

## When not to use Katz Centrality
* in graphs with heavy cyclical dependency (feedback loops), Katz Centrality might not converge preventing usable results.
* when a graph contains multiple distinct influence factors Katz can blur them.
* Katz is very expensive so use in large graphs depends on cuGraph parallelism to be viable.

## How computationally expensive is it?
Katz centraility has several stages with costs that add up as the graph gets larger. The overall cost is often O(n<sup>2</sup>) to O(n<sup>3</sup>) where n is the number of nodes.

___
Copyright (c) 2023, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");  you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
___