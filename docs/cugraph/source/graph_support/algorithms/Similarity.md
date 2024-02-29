
# cuGraph Similarity Notebooks

The RAPIDS cuGraph Similarity folder contain a collection of Jupyter Notebooks that demonstrate algorithms to quantify the similarity between pairs of vertices in the graph.
Results of Similarity algorithms are often used to answer questions like:
* Could two vertices be duplicates or aliases of the same actor?
* Can we predict missing edges based of the similarity between two nodes?
* Are multiple similar communities within the graph?
* Can I create recommendations based on the similarity between vertices in the graph.


Manipulation of the data before or after the graph analytic is not covered here.   Extended, more problem focused, notebooks are being created and available https://github.com/rapidsai/notebooks-extended

## Summary

|Algorithm          |Notebooks Containing                                                     |Description                                                  |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|[Jaccard Smiliarity](./jaccard_similarity.html)| [Jaccard Similarity](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/link_prediction/Jaccard-Similarity.ipynb)                 ||
|[Overlap Similarity](./overlap_similarity.html)| [Overlap Similarity](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/link_prediction/Overlap-Similarity.ipynb)                    ||
|[Sorensen](./sorensen_coefficient.html)|[Sorensen Similarity](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/link_prediction/Sorensen_coefficient.ipynb)||
|Personal Pagerank|[Pagerank](https://github.com/rapidsai/cugraph/blob/main/notebooks/algorithms/link_analysis/Pagerank.ipynb)                 ||


[System Requirements](../../README.md#requirements)

| Author Credit |    Date    |  Update          | cuGraph Version |  Test Hardware |
| --------------|------------|------------------|-----------------|----------------|
| Brad Rees     | 04/19/2021 | created          | 0.19            | GV100, CUDA 11.0
| Don Acosta    | 07/05/2022 | tested / updated | 22.08 nightly   | DGX Tesla V100 CUDA 11.5

## Copyright

Copyright (c) 2019 - 2023, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");  you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
___
