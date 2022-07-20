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
from pylibcugraph._cugraph_c.core_algorithms cimport (   
    cugraph_core_result_t,
    cugraph_core_number,
    cugraph_core_result_get_vertices,
    cugraph_core_result_get_core_numbers,
    cugraph_core_result_free,
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

def EXPERIMENTAL__core_number(ResourceHandle resource_handle,
                              _GPUGraph graph,
                              degree_type,
                              bool_t do_expensive_check):
    """
    Computes core number.

    Parameters
    ----------
    resource_handle: ResourceHandle
        Handle to the underlying device and host resource needed for
        referencing data and running algorithms.
    
    graph: MGGraph
        The input graph, for Multi-GPU operations.
    
    degree_type: device array type
        Device array containing the degree type as a character.
    
    do_expensive_check: bool
        If True, performs more extensive tests on the inputs to ensure
        validity, at the expense of increased run time.

    Returns
    -------
    A tuple of device arrays, where the first item in the tuple is a device
    array containing the vertices and the second item in the tuple is a device
    array containing the core numbers for the corresponding vertices.

    Examples
    --------
    >>> import pylibcugraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 3], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibcugraph.ResourceHandle()
    >>> graph_props = pylibcugraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibcugraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)

    """
    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_core_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    error_code = cugraph_core_number(c_resource_handle_ptr,
                                     c_graph_ptr,
                                     degree_type,
                                     do_expensive_check,
                                     &result_ptr,
                                     &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_core_number")

    cdef cugraph_type_erased_device_array_view_t* vertices_ptr = \
        cugraph_core_result_get_vertices(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* values_ptr = \
        cugraph_core_result_get_core_numbers(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_values = copy_to_cupy_array(c_resource_handle_ptr, values_ptr)

    cugraph_core_result_free(result_ptr)

    return (cupy_vertices, cupy_values)
