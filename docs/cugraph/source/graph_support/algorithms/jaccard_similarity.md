# Jaccard Similarity

The Jaccard similarity between two sets is defined as the ratio of the volume of their intersection divided by the volume of their union. 

The Jaccard Similarity can then be defined as

Jaccard similarity coefficient = $\frac{|A \cap B|}{|A \cup B|}$

In graphs, the sets refer to the set of connected nodes or neighborhood of nodes A and B.

[Learn more about Jaccard Similarity](https://en.wikipedia.org/wiki/Jaccard_index)

## When to use Jaccard Similarity
* You want to find whether two nodes in a graph are in similar communities.
* You want to compare the structure of two graphs.
* You have a set of graphs and want to classify them as particular types

## When not to use Jaccard Similarity
* In directed graphs
* in very large sparse graphs
* Graphs with large disparities in node degrees

## How computationally expensive is it?
While cuGraph's parallelism mitigates run cost, [Big O notation](https://en.wikipedia.org/wiki/Big_O_notation) is still the standard to compare algorithm costs.

The cost of calculating the Jaccard Similarity for a graph is  O(d * n) where d is the average degree of the nodes and n is the number of nodes.

___
Copyright (c) 2023, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");  you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
___