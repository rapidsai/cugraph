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


cdef extern from "cugraph_c/algorithms.h":
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
            const cugraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
            double alpha,
            double epsilon,
            size_t max_iterations,
            bool_t has_initial_guess,
            bool_t do_expensive_check,
            cugraph_centrality_result_t** result,
            cugraph_error_t** error
        )

    cdef cugraph_error_code_t \
        cugraph_personalized_pagerank(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            const cugraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
            cugraph_type_erased_device_array_view_t* personalization_vertices,
            const cugraph_type_erased_device_array_view_t* personalization_values,
            double alpha,
            double epsilon,
            size_t max_iterations,
            bool_t has_initial_guess,
            bool_t do_expensive_check,
            cugraph_centrality_result_t** result,
            cugraph_error_t** error
        )

    ###########################################################################
    # paths and path extraction
    ctypedef struct cugraph_paths_result_t:
        pass

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_paths_result_get_vertices(
            cugraph_paths_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_paths_result_get_distances(
            cugraph_paths_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_paths_result_get_predecessors(
            cugraph_paths_result_t* result
        )

    cdef void \
        cugraph_paths_result_free(
            cugraph_paths_result_t* result
        )

    ctypedef struct cugraph_extract_paths_result_t:
        pass

    cdef cugraph_error_code_t \
        cugraph_extract_paths(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            const cugraph_type_erased_device_array_view_t* sources,
            const cugraph_paths_result_t* paths_result,
            const cugraph_type_erased_device_array_view_t* destinations,
            cugraph_extract_paths_result_t** result,
            cugraph_error_t** error
        )

    cdef size_t \
        cugraph_extract_paths_result_get_max_path_length(
            cugraph_extract_paths_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_extract_paths_result_get_paths(
            cugraph_extract_paths_result_t* result
        )

    cdef void \
        cugraph_extract_paths_result_free(
            cugraph_extract_paths_result_t* result
        )

    ###########################################################################
    # bfs
    cdef cugraph_error_code_t \
        cugraph_bfs(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            # FIXME: this may become const
            cugraph_type_erased_device_array_view_t* sources,
            bool_t direction_optimizing,
            size_t depth_limit,
            bool_t compute_predecessors,
            bool_t do_expensive_check,
            cugraph_paths_result_t** result,
            cugraph_error_t** error
        )

    ###########################################################################
    # sssp
    cdef cugraph_error_code_t \
        cugraph_sssp(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            size_t source,
            double cutoff,
            bool_t compute_predecessors,
            bool_t do_expensive_check,
            cugraph_paths_result_t** result,
            cugraph_error_t** error
        )

    ###########################################################################
    # random_walks
    ctypedef struct cugraph_random_walk_result_t:
        pass

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_random_walk_result_get_paths(
            cugraph_random_walk_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_random_walk_result_get_weights(
            cugraph_random_walk_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_random_walk_result_get_path_sizes(
            cugraph_random_walk_result_t* result
        )

    cdef void \
        cugraph_random_walk_result_free(
            cugraph_random_walk_result_t* result
        )

    # node2vec
    cdef cugraph_error_code_t \
        cugraph_node2vec(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            const cugraph_type_erased_device_array_view_t* sources,
            size_t max_depth,
            bool_t compress_result,
            double p,
            double q,
            cugraph_random_walk_result_t** result,
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

    ###########################################################################
    # sampling
    ctypedef struct cugraph_sample_result_t:
        pass
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_sample_result_get_sources(
            cugraph_sample_result_t* result
        )
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_sample_result_get_destinations(
            cugraph_sample_result_t* result
        )
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_sample_result_get_start_labels(
            cugraph_sample_result_t* result
        )
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_sample_result_get_index(
            cugraph_sample_result_t* result
        )
    
    cdef cugraph_type_erased_host_array_view_t* \
        cugraph_sample_result_get_counts(
            cugraph_sample_result_t* result
        )
    
    cdef void \
        cugraph_sample_result_free(
            cugraph_sample_result_t* result
        )
    
    # uniform neighborhood sampling
    cdef cugraph_error_code_t \
        cugraph_uniform_neighbor_sample(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            const cugraph_type_erased_device_array_view_t* start,
            const cugraph_type_erased_device_array_view_t* start_labels,
            const cugraph_type_erased_host_array_view_t* fan_out,
            bool_t without_replacement,
            bool_t do_expensive_check,
            cugraph_sample_result_t** result,
            cugraph_error_t** error
        )
