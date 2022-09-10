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

import numpy

from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.resource_handle cimport (
    cugraph_resource_handle_t,
)
from pylibcugraph.internal_types cimport SamplingResult
from pylibcugraph._cugraph_c.algorithms cimport (
    cugraph_sample_result_t,
    cugraph_sample_result_create,
    cugraph_sample_result_free,
)
from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)
from pylibcugraph.utils cimport (
    assert_success,
    assert_AI_type,
    get_c_type_from_numpy_type,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_host_array_view_t,
    cugraph_type_erased_host_array_view_create,
    cugraph_type_erased_host_array_view_free,
)


def create_sampling_result(ResourceHandle resource_handle,
    			   sources,
                           destinations,
                           indices):
    """
    Create a SamplingResult object from a cugraph_sample_result_t* typically
    returned by cugraph C sampling algos.
    """
    assert_AI_type(sources, "sources")
    assert_AI_type(destinations, "destinations")
    assert_AI_type(indices, "indices")

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr

    cdef cugraph_sample_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef uintptr_t ai_srcs_ptr = \
        sources.__array_interface__["data"][0]
    cdef uintptr_t ai_dsts_ptr = \
        destinations.__array_interface__["data"][0]
    cdef uintptr_t ai_inds_ptr = \
        indices.__array_interface__["data"][0]
    unused = numpy.ndarray(0, dtype="int32")
    cdef uintptr_t ai_cnts_ptr = \
        unused.__array_interface__["data"][0]

    cdef cugraph_type_erased_host_array_view_t* c_srcs_ptr = \
        cugraph_type_erased_host_array_view_create(
            <void*>ai_srcs_ptr,
            len(sources),
            get_c_type_from_numpy_type(sources.dtype))
    cdef cugraph_type_erased_host_array_view_t* c_dsts_ptr = \
        cugraph_type_erased_host_array_view_create(
            <void*>ai_dsts_ptr,
            len(destinations),
            get_c_type_from_numpy_type(sources.dtype))
    cdef cugraph_type_erased_host_array_view_t* c_inds_ptr = \
        cugraph_type_erased_host_array_view_create(
            <void*>ai_inds_ptr,
            len(indices),
            get_c_type_from_numpy_type(sources.dtype))
    cdef cugraph_type_erased_host_array_view_t* c_cnts_ptr = \
        cugraph_type_erased_host_array_view_create(
            <void*>ai_cnts_ptr,
            len(unused),
            get_c_type_from_numpy_type(unused.dtype))

    error_code = cugraph_sample_result_create(
        c_resource_handle_ptr,
        c_srcs_ptr,
        c_dsts_ptr,
        c_inds_ptr,
        c_cnts_ptr,
        &result_ptr,
        &error_ptr)
    assert_success(error_code, error_ptr, "create_sampling_result")

    cugraph_type_erased_host_array_view_free(c_srcs_ptr)
    cugraph_type_erased_host_array_view_free(c_dsts_ptr)
    cugraph_type_erased_host_array_view_free(c_inds_ptr)
    cugraph_type_erased_host_array_view_free(c_cnts_ptr)
