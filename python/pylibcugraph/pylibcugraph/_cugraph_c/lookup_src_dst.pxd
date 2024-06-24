# Copyright (c) 2024, NVIDIA CORPORATION.
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
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)

cdef extern from "cugraph_c/lookup_src_dst.h":
    ###########################################################################

    ctypedef struct cugraph_lookup_container_t:
       pass

    ctypedef struct cugraph_lookup_result_t:
       pass

    cdef cugraph_error_code_t cugraph_build_edge_id_and_type_to_src_dst_lookup_map(
        const cugraph_resource_handle_t* handle,
        cugraph_graph_t* graph,
        cugraph_lookup_container_t** lookup_container,
        cugraph_error_t** error)

    cdef cugraph_error_code_t cugraph_lookup_endpoints_from_edge_ids_and_single_type(
        const cugraph_resource_handle_t* handle,
        cugraph_graph_t* graph,
        const cugraph_lookup_container_t* lookup_container,
        const cugraph_type_erased_device_array_view_t* edge_ids_to_lookup,
        int edge_type_to_lookup,
        cugraph_lookup_result_t** result,
        cugraph_error_t** error)

    cdef cugraph_error_code_t cugraph_lookup_endpoints_from_edge_ids_and_types(
        const cugraph_resource_handle_t* handle,
        cugraph_graph_t* graph,
        const cugraph_lookup_container_t* lookup_container,
        const cugraph_type_erased_device_array_view_t* edge_ids_to_lookup,
        const cugraph_type_erased_device_array_view_t* edge_types_to_lookup,
        cugraph_lookup_result_t** result,
        cugraph_error_t** error)

    cdef cugraph_type_erased_device_array_view_t* cugraph_lookup_result_get_srcs(
        const cugraph_lookup_result_t* result)

    cdef cugraph_type_erased_device_array_view_t* cugraph_lookup_result_get_dsts(
        const cugraph_lookup_result_t* result)
