# Betweenness Centrality (BC)

Betweenness centrality is a measure of the relative importance based on measuring the number of shortest paths that pass through each vertex or over each edge.  High betweenness centrality vertices have a greater number of path cross through the vertex.  Likewise, high centrality edges have more shortest paths that pass over the edge.

See [Betweenness on Wikipedia](https://en.wikipedia.org/wiki/Betweenness_centrality) for more details on the algorithm.

Betweenness centrality of a node 𝑣 is the sum of the fraction of all-pairs shortest paths that pass through 𝑣

$c_B(v) =\sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}$


## When to use BC
Betweenness centrality is often used to answer questions like:
* Which vertices are most influential in the network?
* What are the bridge vertices in a network?
* How robust/redundant is the network?
* In a social network analysis, betweenness centrality can be used to identify roles in an organization.

## When not to use BC
Betweenness Centrality is less efficient in certain circumstances:
* Large graphs may require approximationing betweenness centrality as the computational cost increases.
* Disconnected networks or networks with many isolated components limit the value of betweenness centrality
* Betweenness centality is more costly and less useful in weighted graphs.
* In networks with hierarchical structure, BC might not accurately reflect true influence
* Networks with multiple edge types often require a seperate method of measuring influence for each edge type.


## How computationally expensive is BC?
While cuGraph's parallelism migigates run time, [Big O notation](https://en.wikipedia.org/wiki/Big_O_notation) is still the standard to compare algorithm costs.
* The cost is  O(V(E+V)) for a non-weighted graph and O(V(E+V)log(V)) for a weighted graph.
* A breadth-first search is done to determine shortest paths betweeb all nodes prior to calculating BC.

## Sample benchmarks
Coming Soon

___
Copyright (c) 2023, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");  you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
___
