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

from libc.stdint cimport uintptr_t

from pylibcugraph._cugraph_c.cugraph_api cimport (
    bool_t,
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_t,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
    cugraph_sg_graph_create,
    cugraph_graph_properties_t,
    cugraph_sg_graph_free,
)


cdef class SGGraph:
    """
    RAII-stye Graph class for use with single-GPU APIs that manages the
    individual create/free calls and the corresponding cugraph_graph_t pointer.
    """
    def __cinit__(self,
                  resource_handle,
                  graph_properties,
                  src_array,
                  dst_array,
                  weight_array,
                  store_transposed,
                  renumber,
                  expensive_check):
        # use __cuda_array_interface__ to get device pointers

        cdef uintptr_t x = 0;
        cdef cugraph_resource_handle_t* rh
        cdef cugraph_graph_properties_t* p
        cdef cugraph_type_erased_device_array_t* a
        cdef cugraph_graph_t* g
        cdef cugraph_error_t* e
        cdef bool_t true=bool_t.TRUE
        cdef bool_t false=bool_t.FALSE

        cugraph_sg_graph_create(rh,
                                p,
                                a,
                                a,
                                a,
                                true,
                                false,
                                false,
                                &g,
                                &e)
        #self.__sg_graph = g

    def __dealloc__(self):
        #cdef cugraph_graph_t* g = self.__sg_graph
        cdef cugraph_graph_t* g
        cugraph_sg_graph_free(g)
