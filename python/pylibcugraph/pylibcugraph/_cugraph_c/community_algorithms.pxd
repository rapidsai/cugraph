# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
    cugraph_induced_subgraph_result_t,
)

from pylibcugraph._cugraph_c.random cimport (
    cugraph_rng_state_t,
)

cdef extern from "cugraph_c/community_algorithms.h":
    ###########################################################################
    # triangle_count
    ctypedef struct cugraph_triangle_count_result_t:
        pass

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_triangle_count_result_get_vertices(
            cugraph_triangle_count_result_t* result
        )
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_triangle_count_result_get_counts(
            cugraph_triangle_count_result_t* result
        )
    
    cdef void \
        cugraph_triangle_count_result_free(
            cugraph_triangle_count_result_t* result
        )
    
    cdef cugraph_error_code_t \
        cugraph_triangle_count(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            const cugraph_type_erased_device_array_view_t* start,
            bool_t do_expensive_check,
            cugraph_triangle_count_result_t** result,
            cugraph_error_t** error
        )

    ###########################################################################
    # louvain
    ctypedef struct cugraph_hierarchical_clustering_result_t:
        pass

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_hierarchical_clustering_result_get_vertices(
            cugraph_hierarchical_clustering_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_hierarchical_clustering_result_get_clusters(
            cugraph_hierarchical_clustering_result_t* result
        )
    
    cdef double cugraph_hierarchical_clustering_result_get_modularity(
        cugraph_hierarchical_clustering_result_t* result
        )

    cdef void \
        cugraph_hierarchical_clustering_result_free(
            cugraph_hierarchical_clustering_result_t* result
        )

    cdef cugraph_error_code_t \
        cugraph_louvain(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            size_t max_level,
            double resolution,
            bool_t do_expensive_check,
            cugraph_hierarchical_clustering_result_t** result,
            cugraph_error_t** error
        )
    
    # extract_ego
    cdef cugraph_error_code_t \
        cugraph_extract_ego(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            const cugraph_type_erased_device_array_view_t* source_vertices,
            size_t radius,
            bool_t do_expensive_check,
            cugraph_induced_subgraph_result_t** result,
            cugraph_error_t** error
        )
    
    # leiden
    ctypedef struct cugraph_hierarchical_clustering_result_t:
        pass

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_hierarchical_clustering_result_get_vertices(
            cugraph_hierarchical_clustering_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_hierarchical_clustering_result_get_clusters(
            cugraph_hierarchical_clustering_result_t* result
        )
    
    cdef double cugraph_hierarchical_clustering_result_get_modularity(
        cugraph_hierarchical_clustering_result_t* result
        )

    cdef void \
        cugraph_hierarchical_clustering_result_free(
            cugraph_hierarchical_clustering_result_t* result
        )

    cdef cugraph_error_code_t \
        cugraph_leiden(
            const cugraph_resource_handle_t* handle,
            cugraph_rng_state_t* rng_state,
            cugraph_graph_t* graph,
            size_t max_level,
            double resolution,
            double theta,
            bool_t do_expensive_check,
            cugraph_hierarchical_clustering_result_t** result,
            cugraph_error_t** error
        )
    ###########################################################################
    # ECG
    cdef cugraph_error_code_t \
        cugraph_ecg(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            double min_weight,
            size_t ensemble_size,
            bool_t do_expensive_check,
            cugraph_hierarchical_clustering_result_t** result,
            cugraph_error_t** error
        )
    
    ###########################################################################
    # Clustering
    ctypedef struct cugraph_clustering_result_t:
        pass
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_clustering_result_get_vertices(
            cugraph_clustering_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_clustering_result_get_clusters(
            cugraph_clustering_result_t* result
        )
    
    cdef void \
        cugraph_clustering_result_free(
            cugraph_clustering_result_t* result
        )

    # Balanced cut clustering
    cdef cugraph_error_code_t \
        cugraph_balanced_cut_clustering(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            size_t n_clusters,
            size_t n_eigenvectors,
            double evs_tolerance,
            int evs_max_iterations,
            double k_means_tolerance,
            int k_means_max_iterations,
            bool_t do_expensive_check,
            cugraph_clustering_result_t** result,
            cugraph_error_t** error
        )
    
    # Spectral modularity maximization
    cdef cugraph_error_code_t \
        cugraph_spectral_modularity_maximization(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            size_t n_clusters,
            size_t n_eigenvectors,
            double evs_tolerance,
            int evs_max_iterations,
            double k_means_tolerance,
            int k_means_max_iterations,
            bool_t do_expensive_check,
            cugraph_clustering_result_t** result,
            cugraph_error_t** error
        )
    
    # Analyze clustering modularity
    cdef cugraph_error_code_t \
        cugraph_analyze_clustering_modularity(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            size_t n_clusters,
            const cugraph_type_erased_device_array_view_t* vertices,
            const cugraph_type_erased_device_array_view_t* clusters,
            double* score,
            cugraph_error_t** error
        )
    
    # Analyze clustering edge cut
    cdef cugraph_error_code_t \
        cugraph_analyze_clustering_edge_cut(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            size_t n_clusters,
            const cugraph_type_erased_device_array_view_t* vertices,
            const cugraph_type_erased_device_array_view_t* clusters,
            double* score,
            cugraph_error_t** error
        )
    
    # Analyze clustering ratio cut
    cdef cugraph_error_code_t \
        cugraph_analyze_clustering_ratio_cut(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            size_t n_clusters,
            const cugraph_type_erased_device_array_view_t* vertices,
            const cugraph_type_erased_device_array_view_t* clusters,
            double* score,
            cugraph_error_t** error
        )

