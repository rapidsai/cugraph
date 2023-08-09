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


cdef extern from "cugraph_c/algorithms.h":
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
    
    cdef size_t \
        cugraph_random_walk_result_get_max_path_length(
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
    # sampling
    ctypedef struct cugraph_sample_result_t:
        pass

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_sample_result_get_renumber_map(
            const cugraph_sample_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_sample_result_get_renumber_map_offsets(
            const cugraph_sample_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_sample_result_get_sources(
            const cugraph_sample_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_sample_result_get_destinations(
            const cugraph_sample_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_sample_result_get_index(
            const cugraph_sample_result_t* result
        )
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_sample_result_get_edge_weight(
            const cugraph_sample_result_t* result
        )
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_sample_result_get_edge_id(
            const cugraph_sample_result_t* result
        )
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_sample_result_get_edge_type(
            const cugraph_sample_result_t* result
        )
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_sample_result_get_hop(
            const cugraph_sample_result_t* result
        )
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_sample_result_get_start_labels(
            const cugraph_sample_result_t* result
        )
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_sample_result_get_offsets(
            const cugraph_sample_result_t* result
        )

    cdef void \
        cugraph_sample_result_free(
            const cugraph_sample_result_t* result
        )

    # testing API - cugraph_sample_result_t instances are normally created only
    # by sampling algos
    cdef cugraph_error_code_t \
        cugraph_test_sample_result_create(
            const cugraph_resource_handle_t* handle,
            const cugraph_type_erased_device_array_view_t* srcs,
            const cugraph_type_erased_device_array_view_t* dsts,
            const cugraph_type_erased_device_array_view_t* edge_id,
            const cugraph_type_erased_device_array_view_t* edge_type,
            const cugraph_type_erased_device_array_view_t* wgt,
            const cugraph_type_erased_device_array_view_t* hop,
            const cugraph_type_erased_device_array_view_t* label,
            cugraph_sample_result_t** result,
            cugraph_error_t** error
        )
    
    ctypedef struct cugraph_sampling_options_t:
        pass
    
    ctypedef enum cugraph_prior_sources_behavior_t:
        DEFAULT
        CARRY_OVER
        EXCLUDE
    
    cdef cugraph_error_code_t \
        cugraph_sampling_options_create(
            cugraph_sampling_options_t** options,
            cugraph_error_t** error,
        )
    
    cdef void \
        cugraph_sampling_set_renumber_results(
            cugraph_sampling_options_t* options,
            bool_t value,
        )

    cdef void \
        cugraph_sampling_set_with_replacement(
            cugraph_sampling_options_t* options,
            bool_t value,
        )
    
    cdef void \
        cugraph_sampling_set_return_hops(
            cugraph_sampling_options_t* options,
            bool_t value,
        )

    cdef void \
        cugraph_sampling_set_prior_sources_behavior(
            cugraph_sampling_options_t* options,
            cugraph_prior_sources_behavior_t value
        )

    cdef void \
        cugraph_sampling_set_dedupe_sources(
            cugraph_sampling_options_t* options,
            bool_t value,
        )
    
    cdef void \
        cugraph_sampling_options_free(
            cugraph_sampling_options_t* options,
    )

    # uniform random walks
    cdef cugraph_error_code_t \
        cugraph_uniform_random_walks(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            const cugraph_type_erased_device_array_view_t* start_vertices,
            size_t max_length,
            cugraph_random_walk_result_t** result,
            cugraph_error_t** error
        )
    
    # biased random walks
    cdef cugraph_error_code_t \
        cugraph_based_random_walks(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            const cugraph_type_erased_device_array_view_t* start_vertices,
            size_t max_length,
            cugraph_random_walk_result_t** result,
            cugraph_error_t** error
        )
