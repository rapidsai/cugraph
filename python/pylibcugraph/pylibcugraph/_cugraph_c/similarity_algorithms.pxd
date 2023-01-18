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
from pylibcugraph._cugraph_c.graph_functions cimport (
    cugraph_vertex_pairs_t
)


cdef extern from "cugraph_c/similarity_algorithms.h":
    ###########################################################################
    #"""
    ctypedef struct cugraph_similarity_result_t:
        pass
    #"""

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_similarity_result_get_similarity(
            cugraph_similarity_result_t* result
        )
    
    cdef void \
        cugraph_similarity_result_free(
            cugraph_similarity_result_t* result
        )
    
    ###########################################################################
    # jaccard coefficients
    cdef cugraph_error_code_t \
        cugraph_jaccard_coefficients(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            const cugraph_vertex_pairs_t* vertex_pairs,
            bool_t use_weight,
            bool_t do_expensive_check,
            cugraph_similarity_result_t** result,
            cugraph_error_t** error
        )

    ###########################################################################
    # sorensen coefficients
    cdef cugraph_error_code_t \
        cugraph_sorensen_coefficients(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            const cugraph_vertex_pairs_t* vertex_pairs,
            bool_t use_weight,
            bool_t do_expensive_check,
            cugraph_similarity_result_t** result,
            cugraph_error_t** error
        )

    ###########################################################################
    # overlap coefficients
    cdef cugraph_error_code_t \
        cugraph_overlap_coefficients(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            const cugraph_vertex_pairs_t* vertex_pairs,
            bool_t use_weight,
            bool_t do_expensive_check,
            cugraph_similarity_result_t** result,
            cugraph_error_t** error
        )
