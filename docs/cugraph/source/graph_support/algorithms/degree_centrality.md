# Degree Centrality
Degree centrality is the simplest measure of the relative importance based on counting the connections with each vertex. Vertices with the most connections are the most central by this measure.

See [Degree Centrality on Wikipedia](https://en.wikipedia.org/wiki/Degree_centrality) for more details on the algorithm.

Degree centrality of a vertex 𝑣 is the sum of the edges incident on that node.

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/62c50cbf5f6cbe0842fe58fca63deb0f0772a829" />

## When to use Degree Centrality
* When you need a really quick identifcation of important nodes on very simply structured data.
* In cases like collaboration networks where all links have equal importance.
* In many biologic and transportation networks, shear number of connections is important to itentify critical nodes whether they be proteins or airports.
* In huge graphs, Degree centrality is a the quickest 

## When not to use Degree Centrality
* When weights, edge direction or edge types matter
* Graphs with self loops
* Multi-graphs ( graphs with multiple edges between the same two nodes)
* In general Degree Centrality falls short in most cases where the data is complex or nuanced.

## How computationally expensive is it?
While cuGraph's parallelism migigates run time, [Big O notation](https://en.wikipedia.org/wiki/Big_O_notation) is still the standard to compare algorithm costs.

The cost of Degree Centrality is O(n) where n is the number of nodes.
___
Copyright (c) 2023, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");  you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
___