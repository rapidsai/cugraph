# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from libc.stdint cimport uintptr_t

import cupy

from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.resource_handle cimport (
    cugraph_resource_handle_t,
)
from pylibcugraph.internal_types.sampling_result cimport SamplingResult
from pylibcugraph._cugraph_c.algorithms cimport (
    cugraph_sample_result_t,
    cugraph_sample_result_free,
)
from pylibcugraph._cugraph_c.sampling_algorithms cimport (
    cugraph_test_uniform_neighborhood_sample_result_create,
)
from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)
from pylibcugraph.utils cimport (
    assert_success,
    assert_device_accessible,
    get_c_type_from_numpy_type,
    get_c_type_from_py_obj,
    get_size_from_py_obj,
    get_data_ptr_from_py_obj,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_device_array_view_create,
    cugraph_type_erased_device_array_view_free,
)


def create_sampling_result(ResourceHandle resource_handle,
		           device_sources,
                           device_destinations,
                           device_weights,
                           device_edge_id,
                           device_edge_type,
                           device_hop,
                           device_batch_label):
    """
    Create a SamplingResult object from individual host arrays.

    This function is currently testing-only because the SamplingResult type is
    considered internal (ie. pylibcugraph users will not be exposed to it) and
    because SamplingResult instances will be created from a
    cugraph_sample_result_t pointer and not host arrays.
    """
    assert_device_accessible(device_sources, "device_sources")
    assert_device_accessible(device_destinations, "device_destinations")
    if device_weights is not None:
        assert_device_accessible(device_weights, "device_weights")
    if device_edge_id is not None:
        assert_device_accessible(device_edge_id, "device_edge_id")
    if device_edge_type is not None:
        assert_device_accessible(device_edge_type, "device_edge_type")
    if device_weights is not None:
        assert_device_accessible(device_weights, "device_weights")
    if device_hop is not None:
        assert_device_accessible(device_hop, "device_hop")
    if device_batch_label is not None:
        assert_device_accessible(device_batch_label, "device_batch_label")

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr

    cdef cugraph_sample_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef uintptr_t cai_srcs_ptr = \
        get_data_ptr_from_py_obj(device_sources)
    cdef uintptr_t cai_dsts_ptr = \
        get_data_ptr_from_py_obj(device_destinations)

    cdef uintptr_t cai_weights_ptr
    if device_weights is not None:
        cai_weights_ptr = get_data_ptr_from_py_obj(device_weights)
    cdef uintptr_t cai_edge_ids_ptr
    if device_edge_id is not None:
        cai_edge_ids_ptr = get_data_ptr_from_py_obj(device_edge_id)
    cdef uintptr_t cai_edge_types_ptr
    if device_edge_type is not None:
        cai_edge_types_ptr = get_data_ptr_from_py_obj(device_edge_type)
    cdef uintptr_t cai_hop_ptr
    if device_hop is not None:
        cai_hop_ptr = get_data_ptr_from_py_obj(device_hop)
    cdef uintptr_t cai_batch_id_ptr
    if device_batch_label is not None:
        cai_batch_id_ptr = get_data_ptr_from_py_obj(device_batch_label)

    cdef cugraph_type_erased_device_array_view_t* c_srcs_view_ptr = (
        cugraph_type_erased_device_array_view_create(
            <void*>cai_srcs_ptr,
            get_size_from_py_obj(device_sources),
            get_c_type_from_py_obj(device_sources))
    )
    cdef cugraph_type_erased_device_array_view_t* c_dsts_view_ptr = (
        cugraph_type_erased_device_array_view_create(
            <void*>cai_dsts_ptr,
            get_size_from_py_obj(device_destinations),
            get_c_type_from_py_obj(device_destinations))
    )
    cdef cugraph_type_erased_device_array_view_t* c_weight_ptr = <cugraph_type_erased_device_array_view_t*>NULL
    if device_weights is not None:
        c_weight_ptr = (
            cugraph_type_erased_device_array_view_create(
                <void*>cai_weights_ptr,
                get_size_from_py_obj(device_weights),
                get_c_type_from_py_obj(device_weights)
            )
        )
    cdef cugraph_type_erased_device_array_view_t* c_edge_id_ptr = <cugraph_type_erased_device_array_view_t*>NULL
    if device_weights is not None:
        c_edge_id_ptr = (
            cugraph_type_erased_device_array_view_create(
                <void*>cai_edge_ids_ptr,
                get_size_from_py_obj(device_edge_id),
                get_c_type_from_py_obj(device_edge_id)
            )
        )
    cdef cugraph_type_erased_device_array_view_t* c_edge_type_ptr = <cugraph_type_erased_device_array_view_t*>NULL
    if device_weights is not None:
        c_edge_type_ptr = (
            cugraph_type_erased_device_array_view_create(
                <void*>cai_edge_types_ptr,
                get_size_from_py_obj(device_edge_type),
                get_c_type_from_py_obj(device_edge_type)
            )
        )

    cdef cugraph_type_erased_device_array_view_t* c_hop_ptr = <cugraph_type_erased_device_array_view_t*>NULL
    if device_weights is not None:
        c_hop_ptr = (
            cugraph_type_erased_device_array_view_create(
                <void*>cai_hop_ptr,
                get_size_from_py_obj(device_hop),
                get_c_type_from_py_obj(device_hop)
            )
        )

    cdef cugraph_type_erased_device_array_view_t* c_label_ptr = <cugraph_type_erased_device_array_view_t*>NULL
    if device_weights is not None:
        c_label_ptr = (
            cugraph_type_erased_device_array_view_create(
                <void*>cai_batch_id_ptr,
                get_size_from_py_obj(device_batch_label),
                get_c_type_from_py_obj(device_batch_label)
            )
        )


    error_code = cugraph_test_uniform_neighborhood_sample_result_create(
        c_resource_handle_ptr,
        c_srcs_view_ptr,
        c_dsts_view_ptr,
        c_edge_id_ptr,
        c_edge_type_ptr,
        c_weight_ptr,
        c_hop_ptr,
        c_label_ptr,
        &result_ptr,
        &error_ptr)
    assert_success(error_code, error_ptr, "create_sampling_result")

    result = SamplingResult()
    result.set_ptr(result_ptr)

    # Free the non-owning view containers. This should not free result data.
    cugraph_type_erased_device_array_view_free(c_srcs_view_ptr)
    cugraph_type_erased_device_array_view_free(c_dsts_view_ptr)
    cugraph_type_erased_device_array_view_free(c_edge_id_ptr)
    cugraph_type_erased_device_array_view_free(c_edge_type_ptr)
    cugraph_type_erased_device_array_view_free(c_weight_ptr)
    cugraph_type_erased_device_array_view_free(c_hop_ptr)
    cugraph_type_erased_device_array_view_free(c_label_ptr)

    return result
