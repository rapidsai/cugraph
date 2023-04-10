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


cdef extern from "cugraph_c/centrality_algorithms.h":
    ###########################################################################
    # pagerank
    ctypedef struct cugraph_centrality_result_t:
        pass

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_centrality_result_get_vertices(
            cugraph_centrality_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_centrality_result_get_values(
            cugraph_centrality_result_t* result
        )

    cdef void \
        cugraph_centrality_result_free(
            cugraph_centrality_result_t* result
        )

    cdef cugraph_error_code_t \
        cugraph_pagerank(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            const cugraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
            const cugraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
            const cugraph_type_erased_device_array_view_t* initial_guess_vertices,
            const cugraph_type_erased_device_array_view_t* initial_guess_values,
            double alpha,
            double epsilon,
            size_t max_iterations,
            bool_t do_expensive_check,
            cugraph_centrality_result_t** result,
            cugraph_error_t** error
        )

    cdef cugraph_error_code_t \
        cugraph_personalized_pagerank(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            const cugraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
            const cugraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
            const cugraph_type_erased_device_array_view_t* initial_guess_vertices,
            const cugraph_type_erased_device_array_view_t* initial_guess_values,
            const cugraph_type_erased_device_array_view_t* personalization_vertices,
            const cugraph_type_erased_device_array_view_t* personalization_values,
            double alpha,
            double epsilon,
            size_t max_iterations,
            bool_t do_expensive_check,
            cugraph_centrality_result_t** result,
            cugraph_error_t** error
        )

    ###########################################################################
    # eigenvector centrality
    cdef cugraph_error_code_t \
        cugraph_eigenvector_centrality(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            double epsilon,
            size_t max_iterations,
            bool_t do_expensive_check,
            cugraph_centrality_result_t** result,
            cugraph_error_t** error
        )
    
    ###########################################################################
    # katz centrality
    cdef cugraph_error_code_t \
        cugraph_katz_centrality(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            const cugraph_type_erased_device_array_view_t* betas,
            double alpha,
            double beta,
            double epsilon,
            size_t max_iterations,
            bool_t do_expensive_check,
            cugraph_centrality_result_t** result,
            cugraph_error_t** error
        )

    ###########################################################################
    # hits
    ctypedef struct cugraph_hits_result_t:
        pass

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_hits_result_get_vertices(
            cugraph_hits_result_t* result
        )
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_hits_result_get_hubs(
            cugraph_hits_result_t* result
        )
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_hits_result_get_authorities(
            cugraph_hits_result_t* result
        )
    
    cdef void \
        cugraph_hits_result_free(
            cugraph_hits_result_t* result
        )

    cdef cugraph_error_code_t \
        cugraph_hits(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            double tol,
            size_t max_iter,
            const cugraph_type_erased_device_array_view_t* initial_hubs_guess_vertices,
            const cugraph_type_erased_device_array_view_t* initial_hubs_guess_values,
            bool_t normalized,
            bool_t do_expensive_check,
            cugraph_hits_result_t** result,
            cugraph_error_t** error
        )
