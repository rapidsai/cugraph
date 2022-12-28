# Copyright (c) 2022, NVIDIA CORPORATION.
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
    cugraph_test_sample_result_create,
    cugraph_sample_result_free,
)
from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)
from pylibcugraph.utils cimport (
    assert_success,
    assert_CAI_type,
    get_c_type_from_numpy_type,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_device_array_view_create,
    cugraph_type_erased_device_array_view_free,
)


def create_sampling_result(ResourceHandle resource_handle,
		           device_sources,
                           device_destinations,
                           device_indices):
    """
    Create a SamplingResult object from individual host arrays.

    This function is currently testing-only because the SamplingResult type is
    considered internal (ie. pylibcugraph users will not be exposed to it) and
    because SamplingResult instances will be created from a
    cugraph_sample_result_t pointer and not host arrays.
    """
    assert_CAI_type(device_sources, "device_sources")
    assert_CAI_type(device_destinations, "device_destinations")
    assert_CAI_type(device_indices, "device_indices")

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr

    cdef cugraph_sample_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef uintptr_t cai_srcs_ptr = \
        device_sources.__cuda_array_interface__["data"][0]
    cdef uintptr_t cai_dsts_ptr = \
        device_destinations.__cuda_array_interface__["data"][0]
    cdef uintptr_t cai_inds_ptr = \
        device_indices.__cuda_array_interface__["data"][0]
    unused = cupy.ndarray(0, dtype="int32")
    cdef uintptr_t cai_cnts_ptr = \
        unused.__cuda_array_interface__["data"][0]

    cdef cugraph_type_erased_device_array_view_t* c_srcs_view_ptr = (
        cugraph_type_erased_device_array_view_create(
            <void*>cai_srcs_ptr,
            len(device_sources),
            get_c_type_from_numpy_type(device_sources.dtype))
    )
    cdef cugraph_type_erased_device_array_view_t* c_dsts_view_ptr = (
        cugraph_type_erased_device_array_view_create(
            <void*>cai_dsts_ptr,
            len(device_destinations),
            get_c_type_from_numpy_type(device_destinations.dtype))
    )
    cdef cugraph_type_erased_device_array_view_t* c_inds_view_ptr = (
        cugraph_type_erased_device_array_view_create(
            <void*>cai_inds_ptr,
            len(device_indices),
            get_c_type_from_numpy_type(device_indices.dtype))
    )
    cdef cugraph_type_erased_device_array_view_t* c_cnts_view_ptr = (
        cugraph_type_erased_device_array_view_create(
            <void*>cai_cnts_ptr,
            len(unused),
            get_c_type_from_numpy_type(unused.dtype))
    )

    error_code = cugraph_test_sample_result_create(
        c_resource_handle_ptr,
        c_srcs_view_ptr,
        c_dsts_view_ptr,
        c_inds_view_ptr,
        c_cnts_view_ptr,
        &result_ptr,
        &error_ptr)
    assert_success(error_code, error_ptr, "create_sampling_result")

    result = SamplingResult()
    result.set_ptr(result_ptr)

    # Free the non-owning view containers. This should not free result data.
    cugraph_type_erased_device_array_view_free(c_srcs_view_ptr)
    cugraph_type_erased_device_array_view_free(c_dsts_view_ptr)
    cugraph_type_erased_device_array_view_free(c_inds_view_ptr)
    cugraph_type_erased_device_array_view_free(c_cnts_view_ptr)

    return result
