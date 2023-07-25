# Copyright (c) 2023, NVIDIA CORPORATION.
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
    cugraph_resource_handle_t,
    cugraph_data_type_id_t,
    bool_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_host_array_view_t,
)
from pylibcugraph._cugraph_c.random cimport (
    cugraph_rng_state_t,
)

cdef extern from "cugraph_c/graph_generators.h":
    ctypedef enum cugraph_generator_distribution_t:
        POWER_LAW
        UNIFORM

    ctypedef struct cugraph_coo_t:
        pass

    ctypedef struct cugraph_coo_list_t:
        pass

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_coo_get_sources(
            cugraph_coo_t* coo
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_coo_get_destinations(
            cugraph_coo_t* coo
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_coo_get_edge_weights(
            cugraph_coo_t* coo
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_coo_get_edge_id(
            cugraph_coo_t* coo
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_coo_get_edge_type(
            cugraph_coo_t* coo
        )

    cdef size_t \
        cugraph_coo_list_size(
            const cugraph_coo_list_t* coo_list
        )

    cdef cugraph_coo_t* \
        cugraph_coo_list_element(
            cugraph_coo_list_t* coo_list,
            size_t index)

    cdef void \
        cugraph_coo_free(
            cugraph_coo_t* coo
        )

    cdef void \
        cugraph_coo_list_free(
            cugraph_coo_list_t* coo_list
        )

    cdef cugraph_error_code_t \
        cugraph_generate_rmat_edgelist(
            const cugraph_resource_handle_t* handle,
            cugraph_rng_state_t* rng_state,
            size_t scale,
            size_t num_edges,
            double a,
            double b,
            double c,
            bool_t clip_and_flip,
            bool_t scramble_vertex_ids,
            cugraph_coo_t** result,
            cugraph_error_t** error
        )

    cdef cugraph_error_code_t \
        cugraph_generate_rmat_edgelists(
            const cugraph_resource_handle_t* handle,
            cugraph_rng_state_t* rng_state,
            size_t n_edgelists,
            size_t min_scale,
            size_t max_scale,
            size_t edge_factor,
            cugraph_generator_distribution_t size_distribution,
            cugraph_generator_distribution_t edge_distribution,
            bool_t clip_and_flip,
            bool_t scramble_vertex_ids,
            cugraph_coo_list_t** result,
            cugraph_error_t** error
        )

    cdef cugraph_error_code_t \
        cugraph_generate_edge_weights(
            const cugraph_resource_handle_t* handle,
            cugraph_rng_state_t* rng_state,
            cugraph_coo_t* coo,
            cugraph_data_type_id_t dtype,
            double minimum_weight,
            double maximum_weight,
            cugraph_error_t** error
        )

    cdef cugraph_error_code_t \
        cugraph_generate_edge_ids(
            const cugraph_resource_handle_t* handle,
            cugraph_coo_t* coo,
            bool_t multi_gpu,
            cugraph_error_t** error
        )

    cdef cugraph_error_code_t \
        cugraph_generate_edge_types(
            const cugraph_resource_handle_t* handle,
            cugraph_rng_state_t* rng_state,
            cugraph_coo_t* coo,
            int min_edge_type,
            int max_edge_type,
            cugraph_error_t** error
        )
