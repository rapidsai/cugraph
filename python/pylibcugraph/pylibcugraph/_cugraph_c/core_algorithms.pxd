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

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.resource_handle cimport (
    bool_t,
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_host_array_view_t,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)


cdef extern from "cugraph_c/core_algorithms.h":
    ###########################################################################
    # core number
    ctypedef struct cugraph_core_result_t:
        pass
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_core_result_get_vertices(
            cugraph_core_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_core_result_get_core_numbers(
            cugraph_core_result_t* result
        )
    
    cdef void \
        cugraph_core_result_free(
            cugraph_core_result_t* result
        )

    ctypedef enum cugraph_k_core_degree_type_t:
        pass

    cdef cugraph_error_code_t \
        cugraph_core_number(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            cugraph_k_core_degree_type_t degree_type,
            bool_t do_expensive_check,
            cugraph_core_result_t** result,
            cugraph_error_t** error
        )