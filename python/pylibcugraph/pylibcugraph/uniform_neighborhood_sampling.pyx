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

from pylibcugraph._cugraph_c.resource_handle cimport (
    bool_t,
    data_type_id_t,
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_device_array_view_create,
    cugraph_type_erased_device_array_free,
    cugraph_type_erased_host_array_view_t,
    cugraph_type_erased_host_array_view_create
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.algorithms cimport (
    cugraph_uniform_neighbor_sample,
    cugraph_sample_result_t,
    cugraph_sample_result_get_sources,
    cugraph_sample_result_get_destinations,
    cugraph_sample_result_get_start_labels,
    cugraph_sample_result_get_index,
    cugraph_sample_result_get_counts,
    cugraph_sample_result_free,
)
from pylibcugraph.resource_handle cimport (
    EXPERIMENTAL__ResourceHandle,
)
from pylibcugraph.graphs cimport (
    _GPUGraph,
    EXPERIMENTAL__MGGraph,
)
from pylibcugraph.utils cimport (
    assert_success,
    copy_to_cupy_array,
    assert_CAI_type,
    assert_AI_type,
    get_c_type_from_numpy_type,
)


def EXPERIMENTAL__uniform_neighborhood_sampling(EXPERIMENTAL__ResourceHandle resource_handle,
                               EXPERIMENTAL__MGGraph input_graph,
                               start_list,
                               labels_list,
                               h_fan_out,
                               bool_t with_replacement,
                               bool_t do_expensive_check):
    """
    Does neighborhood sampling, which samples nodes from a graph based on the
    current node's neighbors, with a corresponding fanout value at each hop.

    Parameters
    ----------
    resource_handle: ResourceHandle
        Handle to the underlying device and host resources needed for
        referencing data and running algorithms.

    input_graph: MGGraph
        The input graph, for Multi-GPU operations.

    start_list: device array type
        Device array containing the list of starting vertices for sampling.

    labels_list: device array type
        Device array containing the starting labels for reorganizing the
        results after sending the input to different callers.

    h_fan_out: numpy array type
        Device array containing the brancing out (fan-out) degrees per
        starting vertex for each hop level.

    with_replacement: bool
        If true, sampling procedure is done with replacement (the same vertex
        can be selected multiple times in the same step).

    do_expensive_check: bool
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    A tuple of device arrays, where the first and second items in the tuple
    are device arrays containing the starting and ending vertices of each
    walk respectively, the third item in the tuple is a device array
    containing the start labels, the fourth item in the tuple is a device
    array containing the indices for reconstructing paths.

    """
    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = input_graph.c_graph_ptr

    assert_CAI_type(start_list, "start_list")
    assert_CAI_type(labels_list, "labels_list")
    assert_AI_type(h_fan_out, "h_fan_out")

    cdef cugraph_sample_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef uintptr_t cai_start_ptr = \
        start_list.__cuda_array_interface__["data"][0]
    cdef uintptr_t cai_labels_ptr = \
        labels_list.__cuda_array_interface__["data"][0]
    cdef uintptr_t ai_fan_out_ptr = \
        h_fan_out.__array_interface__["data"][0]

    cdef cugraph_type_erased_device_array_view_t* start_ptr = \
        cugraph_type_erased_device_array_view_create(
            <void*>cai_start_ptr,
            len(start_list),
            get_c_type_from_numpy_type(start_list.dtype))
    cdef cugraph_type_erased_device_array_view_t* start_labels_ptr = \
        cugraph_type_erased_device_array_view_create(
            <void*>cai_labels_ptr,
            len(labels_list),
            get_c_type_from_numpy_type(labels_list.dtype))
    cdef cugraph_type_erased_host_array_view_t* fan_out_ptr = \
        cugraph_type_erased_host_array_view_create(
            <void*>ai_fan_out_ptr,
            len(h_fan_out),
            get_c_type_from_numpy_type(h_fan_out.dtype))

    error_code = cugraph_uniform_neighbor_sample(c_resource_handle_ptr,
                                                 c_graph_ptr,
                                                 start_ptr,
                                                 start_labels_ptr,
                                                 fan_out_ptr,
                                                 with_replacement,
                                                 do_expensive_check,
                                                 &result_ptr,
                                                 &error_ptr)
    assert_success(error_code, error_ptr, "uniform_nbr_sample")

    # TODO: counts is a part of the output, but another copy_to_cupy array
    # with appropriate host array types would likely be required. Also
    # potential memory leak until this is covered
    cdef cugraph_type_erased_device_array_view_t* src_ptr = \
        cugraph_sample_result_get_sources(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* dst_ptr = \
        cugraph_sample_result_get_destinations(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* labels_ptr = \
        cugraph_sample_result_get_start_labels(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* index_ptr = \
        cugraph_sample_result_get_index(result_ptr)
    # cdef cugraph_type_erased_host_array_view_t* counts_ptr = \
    #    cugraph_sample_result_get_counts(result_ptr)

    cupy_sources = copy_to_cupy_array(c_resource_handle_ptr, src_ptr)
    cupy_destinations = copy_to_cupy_array(c_resource_handle_ptr, dst_ptr)
    cupy_labels = copy_to_cupy_array(c_resource_handle_ptr, labels_ptr)
    cupy_indices = copy_to_cupy_array(c_resource_handle_ptr, index_ptr)
    # cupy_counts = copy_to_cupy_array(c_resource_handle_ptr, counts_ptr)

    return (cupy_sources, cupy_destinations, cupy_labels, cupy_indices)
    # return (cupy_sources, cupy_destinations, cupy_labels, cupy_indices, cupy_counts)
