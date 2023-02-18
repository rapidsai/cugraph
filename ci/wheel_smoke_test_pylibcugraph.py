# Copyright (c) 2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy
from pylibcugraph import ResourceHandle, GraphProperties, SGGraph, pagerank

# an import statement that reveals a problem with cusolver
from pylibcugraph.components._connectivity import (
    strongly_connected_components,
)


if __name__ == "__main__":
    src_array = cupy.asarray([100, 201, 302], dtype="int32")
    dst_array = cupy.asarray([201, 302, 403], dtype="int32")
    wgt_array = cupy.asarray([1.0, 1.0, 1.0], dtype="float32")

    resource_handle = ResourceHandle()

    G = SGGraph(resource_handle,
                GraphProperties(is_symmetric=False, is_multigraph=False),
                src_array,
                dst_array,
                wgt_array,
                store_transposed=True,
                renumber=True,
                do_expensive_check=True,
                )

    (vertices, pageranks) = pagerank(resource_handle=resource_handle,
                                     graph=G,
                                     precomputed_vertex_out_weight_vertices=None,
                                     precomputed_vertex_out_weight_sums=None,
                                     initial_guess_vertices=None,
                                     initial_guess_values=None,
                                     alpha=0.85,
                                     epsilon=1.0e-6,
                                     max_iterations=500,
                                     do_expensive_check=True,
                                     )

    assert(pageranks.sum() == 1.0)
    results = dict(zip(vertices.tolist(),pageranks.tolist()))
    assert(results[100] < results[201] < results[302] < results[403])
