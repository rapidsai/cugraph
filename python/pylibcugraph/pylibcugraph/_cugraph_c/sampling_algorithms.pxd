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
from pylibcugraph._cugraph_c.algorithms cimport (
    cugraph_sample_result_t,
)
from pylibcugraph._cugraph_c.random cimport (
    cugraph_rng_state_t,
)

cdef extern from "cugraph_c/sampling_algorithms.h":
    ###########################################################################
    cdef cugraph_error_code_t cugraph_uniform_neighbor_sample_with_edge_properties(
        const cugraph_resource_handle_t* handle,
        cugraph_graph_t* graph,
        const cugraph_type_erased_device_array_view_t* start,
        const cugraph_type_erased_device_array_view_t* label,
        const cugraph_type_erased_host_array_view_t* fan_out,
        cugraph_rng_state_t* rng_state,
        bool_t with_replacement,
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