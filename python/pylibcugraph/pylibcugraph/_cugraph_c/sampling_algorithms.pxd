# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
from pylibcugraph._cugraph_c.algorithms cimport (
    cugraph_sample_result_t,
    cugraph_sampling_options_t,
)
from pylibcugraph._cugraph_c.random cimport (
    cugraph_rng_state_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_t,
)
from pylibcugraph._cugraph_c.coo cimport (
    cugraph_coo_t,
)
from pylibcugraph._cugraph_c.properties cimport (
    cugraph_edge_property_view_t,
)

cdef extern from "cugraph_c/sampling_algorithms.h":
    ###########################################################################

    cdef cugraph_error_code_t cugraph_uniform_neighbor_sample(
        const cugraph_resource_handle_t* handle,
        cugraph_graph_t* graph,
        const cugraph_type_erased_device_array_view_t* start_vertices,
        const cugraph_type_erased_device_array_view_t* start_vertex_labels,
        const cugraph_type_erased_device_array_view_t* label_list,
        const cugraph_type_erased_device_array_view_t* label_to_comm_rank,
        const cugraph_type_erased_device_array_view_t* label_offsets,
        const cugraph_type_erased_host_array_view_t* fan_out,
        cugraph_rng_state_t* rng_state,
        const cugraph_sampling_options_t* options,
        bool_t do_expensive_check,
        cugraph_sample_result_t** result,
        cugraph_error_t** error
    )

    cdef cugraph_error_code_t cugraph_biased_neighbor_sample(
        const cugraph_resource_handle_t* handle,
        cugraph_graph_t* graph,
        const cugraph_edge_property_view_t* edge_biases,
        const cugraph_type_erased_device_array_view_t* start_vertices,
        const cugraph_type_erased_device_array_view_t* start_vertex_labels,
        const cugraph_type_erased_device_array_view_t* label_list,
        const cugraph_type_erased_device_array_view_t* label_to_comm_rank,
        const cugraph_type_erased_device_array_view_t* label_offsets,
        const cugraph_type_erased_host_array_view_t* fan_out,
        cugraph_rng_state_t* rng_state,
        const cugraph_sampling_options_t* options,
        bool_t do_expensive_check,
        cugraph_sample_result_t** result,
        cugraph_error_t** error
    )

    cdef cugraph_error_code_t cugraph_test_uniform_neighborhood_sample_result_create(
        const cugraph_resource_handle_t* handle,
        const cugraph_type_erased_device_array_view_t* srcs,
        const cugraph_type_erased_device_array_view_t* dsts,
        const cugraph_type_erased_device_array_view_t* edge_id,
        const cugraph_type_erased_device_array_view_t* edge_type,
        const cugraph_type_erased_device_array_view_t* weight,
        const cugraph_type_erased_device_array_view_t* hop,
        const cugraph_type_erased_device_array_view_t* label,
        cugraph_sample_result_t** result,
        cugraph_error_t** error
    )

    # random vertices selection
    cdef cugraph_error_code_t \
        cugraph_select_random_vertices(
            const cugraph_resource_handle_t* handle,
            const cugraph_graph_t* graph,
            cugraph_rng_state_t* rng_state,
            size_t num_vertices,
            cugraph_type_erased_device_array_t** vertices,
            cugraph_error_t** error
        )

    # negative sampling
    cdef cugraph_error_code_t \
        cugraph_negative_sampling(
            const cugraph_resource_handle_t* handle,
            cugraph_rng_state_t* rng_state,
            cugraph_graph_t* graph,
            size_t num_samples,
            const cugraph_type_erased_device_array_view_t* vertices,
            const cugraph_type_erased_device_array_view_t* src_bias,
            const cugraph_type_erased_device_array_view_t* dst_bias,
            bool_t remove_duplicates,
            bool_t remove_false_negatives,
            bool_t exact_number_of_samples,
            bool_t do_expensive_check,
            cugraph_coo_t **result,
            cugraph_error_t **error
        )
