# Vertex Similarity
----

In this folder we will explore and compare the various vertex similarity metrics available in cuGraph.  [Vertex similarity](https://en.wikipedia.org/wiki/Similarity_(network_science)), as the name implies, is a measure how similar two vertices are.  

|Algorithm          |Notebooks Containing                                                     |Description                                                  |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|Jaccard Similarity  | [Jaccard](Jaccard-Similarity.ipynb)   | Uses the ratio of the volume of vertex intersection divided by the volume of union to determine similarity  |
|Jaccard Weighted Similarity  | [Jaccard](Jaccard-Similarity.ipynb)   | Adds weights to the standard Jaccard Similarity  |
|Overlap Similarity  | [Overlap](Overlap-Similarity.ipynb)               | Evaluates the neighborhood of vertex pairs and looks at the number of common neighbors|

Currently, cuGraph supports the following similarity metrics:
- Jaccard Similarity (also called the Jaccard Index)
- Overlap Coefficient
- Weight Jaccard

Similarity can be between neighboring vertices (default) or second hop neighbors

## Introduction - Overlap (Common Neighbor) Similarity

One of the most common types of vertex similarity is to evaluate the neighborhood of vertex pairs and looks at the number of common neighbors.  That type of similarity comes from statistics and is based on set comparison.  Both Jaccard and the Overlap Coefficient operate on sets, and in a graph setting, those sets are the list of neighboring vertices. <br>
For those that like math:  The neighbors of a vertex, _v_, is defined as the set, _U_, of vertices connected by way of an edge to vertex v, or _N(v) = {U} where v ∈ V and ∀ u ∈ U ∃ edge(v,u)∈ E_.

For the rest of this introduction, set __A__ will equate to _A = N(i)_ and set __B__ will equate to _B = N(j)_.  That just make the rest of the text more readable.

### Additional Reading
- [Similarity in graphs: Jaccard versus the Overlap Coefficient](https://medium.com/rapids-ai/similarity-in-graphs-jaccard-versus-the-overlap-coefficient-610e083b877d)
- [Wikipedia: Jaccard](https://en.wikipedia.org/wiki/Jaccard_index)
- [Wikipedia: Overlap Coefficient](https://en.wikipedia.org/wiki/Overlap_coefficient)

## Copyright

Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");  you may not use this file except in compliance with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.

![RAPIDS](../../img/rapids_logo.png)
