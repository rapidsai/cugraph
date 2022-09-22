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
    cugraph_type_erased_device_array_view_free,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.community_algorithms cimport (
    cugraph_triangle_count_result_t,
    cugraph_triangle_count,
    cugraph_triangle_count_result_get_vertices,
    cugraph_triangle_count_result_get_counts,
    cugraph_triangle_count_result_free,
)
from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)
from pylibcugraph.graphs cimport (
    _GPUGraph,
)
from pylibcugraph.utils cimport (
    assert_success,
    copy_to_cupy_array,
    assert_CAI_type,
    get_c_type_from_numpy_type,
)


def triangle_count(ResourceHandle resource_handle,
                   _GPUGraph graph,
                   start_list,
                   bool_t do_expensive_check):
    """
    Computes the number of triangles (cycles of length three) and the number
    per vertex in the input graph.

    Parameters
    ----------
    resource_handle: ResourceHandle
        Handle to the underlying device and host resources needed for
        referencing data and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    start_list: device array type
        Device array containing the list of vertices for triangle counting.
        If 'None' the entire set of vertices in the graph is processed
    
    do_expensive_check: bool
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.
    
    Returns
    -------
    A tuple of device arrays, where the first item in the tuple is a device
    array containing the vertex identifiers and the second item contains the
    triangle counting counts

    Examples
    --------
    # FIXME: No example yet

    """
    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    assert_CAI_type(start_list, "start_list", allow_None=True)

    cdef cugraph_triangle_count_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef uintptr_t cai_start_ptr
    cdef cugraph_type_erased_device_array_view_t* start_ptr

    if start_list is not None:
        cai_start_ptr = start_list.__cuda_array_interface__["data"][0]
        start_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_start_ptr,
                len(start_list),
                get_c_type_from_numpy_type(start_list.dtype))
    else:
        start_ptr = NULL
    
    error_code = cugraph_triangle_count(c_resource_handle_ptr,
                                        c_graph_ptr,
                                        start_ptr,
                                        do_expensive_check,
                                        &result_ptr,
                                        &error_ptr)
    assert_success(error_code, error_ptr, "triangle_count")

    cdef cugraph_type_erased_device_array_view_t* vertices_ptr = \
        cugraph_triangle_count_result_get_vertices(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* counts_ptr = \
        cugraph_triangle_count_result_get_counts(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_counts = copy_to_cupy_array(c_resource_handle_ptr, counts_ptr)

    cugraph_triangle_count_result_free(result_ptr)

    if start_list is not None:
        cugraph_type_erased_device_array_view_free(start_ptr)
    
    return (cupy_vertices, cupy_counts)
