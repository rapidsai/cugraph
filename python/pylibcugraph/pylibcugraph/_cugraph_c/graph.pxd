# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.types cimport (
    bool_t,
)
from pylibcugraph._cugraph_c.resource_handle cimport (
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

    cdef cugraph_error_code_t \
        cugraph_graph_create_with_times_sg(
            const cugraph_resource_handle_t* handle,
            const cugraph_graph_properties_t* properties,
            const cugraph_type_erased_device_array_view_t* vertices,
            const cugraph_type_erased_device_array_view_t* src,
            const cugraph_type_erased_device_array_view_t* dst,
            const cugraph_type_erased_device_array_view_t* weights,
            const cugraph_type_erased_device_array_view_t* edge_ids,
            const cugraph_type_erased_device_array_view_t* edge_type_ids,
            const cugraph_type_erased_device_array_view_t* edge_start_time_ids,
            const cugraph_type_erased_device_array_view_t* edge_end_time_ids,
            bool_t store_transposed,
            bool_t renumber,
            bool_t drop_self_loops,
            bool_t drop_multi_edges,
            bool_t symmetrize,
            bool_t do_expensive_check,
            cugraph_graph_t** graph,
            cugraph_error_t** error);

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

    cdef cugraph_error_code_t \
        cugraph_graph_create_with_times_mg(
            const cugraph_resource_handle_t * handle,
            const cugraph_graph_properties_t * properties,
            const cugraph_type_erased_device_array_view_t* const* vertices,
            const cugraph_type_erased_device_array_view_t* const* src,
            const cugraph_type_erased_device_array_view_t* const* dst,
            const cugraph_type_erased_device_array_view_t* const* weights,
            const cugraph_type_erased_device_array_view_t* const* edge_ids,
            const cugraph_type_erased_device_array_view_t* const* edge_type_ids,
            const cugraph_type_erased_device_array_view_t* const* edge_start_time_ids,
            const cugraph_type_erased_device_array_view_t* const* edge_end_time_ids,
            bool_t store_transposed,
            size_t num_arrays,
            bool_t drop_self_loops,
            bool_t drop_multi_edges,
            bool_t symmetrize,
            bool_t do_expensive_check,
            cugraph_graph_t** graph,
            cugraph_error_t** error)
