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


cdef extern from "cugraph_c/graph.h":

    ctypedef struct cugraph_graph_t:
        pass

    ctypedef struct cugraph_graph_properties_t:
        bool_t is_symmetric
        bool_t is_multigraph

    # Supports isolated vertices
    cdef cugraph_error_code_t \
         cugraph_graph_create_sg(
             const cugraph_resource_handle_t* handle,
             const cugraph_graph_properties_t* properties,
             const cugraph_type_erased_device_array_view_t* vertices,
             const cugraph_type_erased_device_array_view_t* src,
             const cugraph_type_erased_device_array_view_t* dst,
             const cugraph_type_erased_device_array_view_t* weights,
             const cugraph_type_erased_device_array_view_t* edge_ids,
             const cugraph_type_erased_device_array_view_t* edge_types,
             bool_t store_transposed,
             bool_t renumber,
             bool_t drop_self_loops,
             bool_t drop_multi_edges,
             bool_t symmetrize,
             bool_t check,
             cugraph_graph_t** graph,
             cugraph_error_t** error)

    # FIXME: Might want to delete 'cugraph_sg_graph_free' and replace
    # 'cugraph_mg_graph_free' by 'cugraph_graph_free'
    cdef void \
        cugraph_graph_free(
            cugraph_graph_t* graph
        )

    cdef cugraph_error_code_t \
        cugraph_graph_create_sg_from_csr(
            const cugraph_resource_handle_t* handle,
            const cugraph_graph_properties_t* properties,
            const cugraph_type_erased_device_array_view_t* offsets,
            const cugraph_type_erased_device_array_view_t* indices,
            const cugraph_type_erased_device_array_view_t* weights,
            const cugraph_type_erased_device_array_view_t* edge_ids,
            const cugraph_type_erased_device_array_view_t* edge_type_ids,
            bool_t store_transposed,
            bool_t renumber,
            bool_t symmetrize,
            bool_t check,
            cugraph_graph_t** graph,
            cugraph_error_t** error
        )

    cdef cugraph_error_code_t \
        cugraph_graph_create_mg(
            const cugraph_resource_handle_t* handle,
            const cugraph_graph_properties_t* properties,
            const cugraph_type_erased_device_array_view_t** vertices,
            const cugraph_type_erased_device_array_view_t** src,
            const cugraph_type_erased_device_array_view_t** dst,
            const cugraph_type_erased_device_array_view_t** weights,
            const cugraph_type_erased_device_array_view_t** edge_ids,
            const cugraph_type_erased_device_array_view_t** edge_type_ids,
            bool_t store_transposed,
            size_t num_arrays,
            bool_t drop_self_loops,
            bool_t drop_multi_edges,
            bool_t symmetrize,
            bool_t do_expensive_check,
            cugraph_graph_t** graph,
            cugraph_error_t** error)
