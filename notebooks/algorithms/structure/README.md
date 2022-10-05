
# cuGraph Structure Algorithms

<img src="../../img/zachary_black_lines.png" width="35%"/>

cuGraph Structure notebooks contain Jupyter Notebooks that demonstrate graph manipulations which support other cuGraph algorithms. Many cuGraph algorithms expect vertices ids formated as a contiguous list of integers. Some only support a directed graph. CuGraph structure algorithms encapsulate that functionality and make all those relying on them more efficient and independent of this aspect graph standardizing.

## Summary

|Algorithm          |Notebooks Containing                                                     |Description                                                  |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|Renumber  | [Renumber](Renumber.ipynb)   | Converts a graph with arbitrary vertex ids into a contiguous series of integers for efficient handling by many other cuGraph algorithms |
|Renumber  | [Renumber2](Renumber-2.ipynb)               | Demonstrates how the renumber function can optimize graph processing by converting the underlying sparse matrix into an edgelist with a much small memory footprint. |
|Symmetrize | [Symmetrize](Symmetrize.ipynb)               |Demonstrates the functionality to transform an undirected graph into a directed graph with edges in each direction as needed for many other cuGraph algorithms.|


[System Requirements](../../README.md#requirements)

| Author Credit |    Date    |  Update          | cuGraph Version |  Test Hardware |
| --------------|------------|------------------|-----------------|----------------|
| Brad Rees     | 04/19/2021 | created          | 0.19            | GV100, CUDA 11.0
| Don Acosta    | 08/29/2022 | tested / updated | 22.08 nightly   | DGX Tesla V100 CUDA 11.5|

## Copyright

Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");  you may not use this file except in compliance with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.

![RAPIDS](../../img/rapids_logo.png)
