# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)


cdef extern from "cugraph_c/core_algorithms.h":
    ###########################################################################
    # core number
    ctypedef struct cugraph_core_result_t:
        pass

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_core_result_get_vertices(
            cugraph_core_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_core_result_get_core_numbers(
            cugraph_core_result_t* result
        )

    cdef void \
        cugraph_core_result_free(
            cugraph_core_result_t* result
        )

    ctypedef enum cugraph_k_core_degree_type_t:
        K_CORE_DEGREE_TYPE_IN=0,
        K_CORE_DEGREE_TYPE_OUT=1,
        K_CORE_DEGREE_TYPE_INOUT=2

    cdef cugraph_error_code_t \
        cugraph_core_number(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            cugraph_k_core_degree_type_t degree_type,
            bool_t do_expensive_check,
            cugraph_core_result_t** result,
            cugraph_error_t** error
        )

    ###########################################################################
    # k-core
    ctypedef struct cugraph_k_core_result_t:
        pass

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_k_core_result_get_src_vertices(
            cugraph_k_core_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_k_core_result_get_dst_vertices(
            cugraph_k_core_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_k_core_result_get_weights(
            cugraph_k_core_result_t* result
        )

    cdef void \
        cugraph_k_core_result_free(
            cugraph_k_core_result_t* result
        )

    cdef cugraph_error_code_t \
        cugraph_core_result_create(
            const cugraph_resource_handle_t* handle,
            cugraph_type_erased_device_array_view_t* vertices,
            cugraph_type_erased_device_array_view_t* core_numbers,
            cugraph_core_result_t** core_result,
            cugraph_error_t** error
        )

    cdef cugraph_error_code_t \
        cugraph_k_core(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            size_t k,
            cugraph_k_core_degree_type_t degree_type,
            const cugraph_core_result_t* core_result,
            bool_t do_expensive_check,
            cugraph_k_core_result_t** result,
            cugraph_error_t** error
        )
