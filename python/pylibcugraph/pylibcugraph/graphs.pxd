# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
    cugraph_type_erased_device_array_view_t,
)
from pylibcugraph._cugraph_c.types cimport (
    cugraph_data_type_id_t)


# Base class allowing functions to accept either SGGraph or MGGraph
# This is not visible in python
cdef class _GPUGraph:
    cdef cugraph_data_type_id_t vertex_type
    cdef cugraph_graph_t* c_graph_ptr
    cdef cugraph_type_erased_device_array_view_t* edge_id_view_ptr
    cdef cugraph_type_erased_device_array_view_t** edge_id_view_ptr_ptr
    cdef cugraph_type_erased_device_array_view_t* weights_view_ptr
    cdef cugraph_type_erased_device_array_view_t** weights_view_ptr_ptr

cdef class SGGraph(_GPUGraph):
    pass

cdef class MGGraph(_GPUGraph):
    pass
