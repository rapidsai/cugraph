# Copyright (c) 2022, NVIDIA CORPORATION.
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

from pylibcugraph.utilities.experimental import experimental
from pylibcugraph._cugraph_c._graph import cugraph_sg_graph_create
from pylibcugraph._cugraph_c._resource_handle import ResourceHandle


@experimental(ns_name="")
class __GPUGraphData:
    """
    Data that resides on one or more GPUs that represents a graph.

    GPUGraphData instances define the mapping between user-provided data that
    represents a graph (vertex IDs, edge lists) and the corresponding
    representation needed to support GPU-based graph operations (0-based vertex
    array indices, CSR, CSC, etc.).

    Deleting a GPUGraphData intance frees the GPU memory used for storing the
    graph, but does not delete the src, dst, and weights arrays used to
    construct the instance.
    """

    def __init__(self, src_array, dst_array, weight_array,
                 store_transposed=False,
                 resource_handle=None):
        self.__graph = None
        if resource_handle is None:
            self.__resource_handle = ResourceHandle()
        else:
            self.__resource_handle = resource_handle

        cugraph_sg_graph_create(self.__resource_handle,
                                None,None,None,None,None,None,None)
