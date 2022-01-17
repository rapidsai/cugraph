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

from pylibcugraph._cugraph_c._cugraph_api cimport (
    bool_t,
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c._error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c._array cimport (
    cugraph_type_erased_device_array_t,
)


cdef extern from "cugraph_c/algorithms.h":

    ctypedef struct cugraph_pagerank_result_t:
        pass

    # ctypedef struct cugraph_paths_result_t:
    #     pass

    # ctypedef struct cugraph_extract_paths_result_t:
    #     pass

    cdef cugraph_type_erased_device_array_t*
        cugraph_pagerank_result_get_vertices(
            cugraph_pagerank_result_t* result
        )

    cdef cugraph_type_erased_device_array_t*
        cugraph_pagerank_result_get_pageranks(
            cugraph_pagerank_result_t* result
        )

    cdef void
        cugraph_pagerank_result_free(
            cugraph_pagerank_result_t* result
        )

    cdef cugraph_error_code_t
        cugraph_pagerank(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            const cugraph_type_erased_device_array_t* precomputed_vertex_out_weight_sums,
            double alpha,
            double epsilon,
            size_t max_iterations,
            bool_t has_initial_guess,
            bool_t do_expensive_check,
            cugraph_pagerank_result_t** result,
            cugraph_error_t** error
        )

    cdef cugraph_error_code_t
        cugraph_personalized_pagerank(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            const cugraph_type_erased_device_array_t* precomputed_vertex_out_weight_sums,
            cugraph_type_erased_device_array_t* personalization_vertices,
            const cugraph_type_erased_device_array_t* personalization_values,
            double alpha,
            double epsilon,
            size_t max_iterations,
            bool_t has_initial_guess,
            bool_t do_expensive_check,
            cugraph_pagerank_result_t** result,
            cugraph_error_t** error
        )
