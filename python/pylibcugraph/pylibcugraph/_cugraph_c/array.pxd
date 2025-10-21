# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.types cimport (
    cugraph_data_type_id_t,
    byte_t,
)
from pylibcugraph._cugraph_c.resource_handle cimport (
    cugraph_resource_handle_t,
)


cdef extern from "cugraph_c/array.h":

    ctypedef struct cugraph_type_erased_device_array_t:
        pass

    ctypedef struct cugraph_type_erased_device_array_view_t:
        pass

    ctypedef struct cugraph_type_erased_host_array_t:
        pass

    ctypedef struct cugraph_type_erased_host_array_view_t:
        pass

    cdef cugraph_error_code_t \
        cugraph_type_erased_device_array_create(
            const cugraph_resource_handle_t* handle,
            cugraph_data_type_id_t dtype,
            size_t n_elems,
            cugraph_type_erased_device_array_t** array,
            cugraph_error_t** error
        )

    cdef void \
        cugraph_type_erased_device_array_free(
            cugraph_type_erased_device_array_t* p
        )

    # cdef void* \
    #     cugraph_type_erased_device_array_release(
    #         cugraph_type_erased_device_array_t* p
    #     )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_type_erased_device_array_view(
            cugraph_type_erased_device_array_t* array
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_type_erased_device_array_view_create(
            void* pointer,
            size_t n_elems,
            cugraph_data_type_id_t dtype
        )

    cdef void \
        cugraph_type_erased_device_array_view_free(
            cugraph_type_erased_device_array_view_t* p
        )

    cdef size_t \
        cugraph_type_erased_device_array_view_size(
            const cugraph_type_erased_device_array_view_t* p
        )

    cdef cugraph_data_type_id_t \
        cugraph_type_erased_device_array_view_type(
            const cugraph_type_erased_device_array_view_t* p
        )

    cdef const void* \
        cugraph_type_erased_device_array_view_pointer(
            const cugraph_type_erased_device_array_view_t* p
        )

    cdef cugraph_error_code_t \
        cugraph_type_erased_host_array_create(
            const cugraph_resource_handle_t* handle,
            cugraph_data_type_id_t dtype,
            size_t n_elems,
            cugraph_type_erased_host_array_t** array,
            cugraph_error_t** error
        )

    cdef void \
        cugraph_type_erased_host_array_free(
            cugraph_type_erased_host_array_t* p
        )

    # cdef void* \
    #     cugraph_type_erased_host_array_release(
    #         cugraph_type_erased_host_array_t* p
    #     )

    cdef cugraph_type_erased_host_array_view_t* \
        cugraph_type_erased_host_array_view(
            cugraph_type_erased_host_array_t* array
        )

    cdef cugraph_type_erased_host_array_view_t* \
        cugraph_type_erased_host_array_view_create(
            void* pointer,
            size_t n_elems,
            cugraph_data_type_id_t dtype
        )

    cdef void \
        cugraph_type_erased_host_array_view_free(
            cugraph_type_erased_host_array_view_t* p
        )

    cdef size_t \
        cugraph_type_erased_host_array_size(
            const cugraph_type_erased_host_array_t* p
        )

    cdef cugraph_data_type_id_t \
        cugraph_type_erased_host_array_type(
            const cugraph_type_erased_host_array_t* p
        )

    cdef void* \
        cugraph_type_erased_host_array_pointer(
            const cugraph_type_erased_host_array_view_t* p
        )

    # cdef void* \
    #    cugraph_type_erased_host_array_view_copy(
    #        const cugraph_resource_handle_t* handle,
    #        cugraph_type_erased_host_array_view_t* dst,
    #        const cugraph_type_erased_host_array_view_t* src,
    #        cugraph_error_t** error
    #    )

    cdef cugraph_error_code_t \
        cugraph_type_erased_device_array_view_copy_from_host(
            const cugraph_resource_handle_t* handle,
            cugraph_type_erased_device_array_view_t* dst,
            const byte_t* h_src,
            cugraph_error_t** error
        )

    cdef cugraph_error_code_t \
        cugraph_type_erased_device_array_view_copy_to_host(
            const cugraph_resource_handle_t* handle,
            byte_t* h_dst,
            const cugraph_type_erased_device_array_view_t* src,
            cugraph_error_t** error
        )

    cdef cugraph_error_code_t \
        cugraph_type_erased_device_array_view_copy(
            const cugraph_resource_handle_t* handle,
            cugraph_type_erased_device_array_view_t* dst,
            const cugraph_type_erased_device_array_view_t* src,
            cugraph_error_t** error
        )
