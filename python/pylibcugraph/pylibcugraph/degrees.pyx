# Copyright (c) 2024, NVIDIA CORPORATION.
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
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.graph_functions cimport (
    cugraph_degrees_result_t,
    cugraph_degrees,
    cugraph_in_degrees,
    cugraph_out_degrees,
    cugraph_degrees_result_get_vertices,
    cugraph_degrees_result_get_in_degrees,
    cugraph_degrees_result_get_out_degrees,
    cugraph_degrees_result_free,
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
    create_cugraph_type_erased_device_array_view_from_py_obj,
)


def in_degrees(ResourceHandle resource_handle,
               _GPUGraph graph,
               source_vertices,
               bool_t do_expensive_check):
    """
    Compute the in degrees for the nodes of the graph.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    source_vertices : cupy array
        The nodes for which we will compute degrees.

    do_expensive_check : bool_t
        A flag to run expensive checks for input arguments if True.

    Returns
    -------
    A tuple of device arrays, where the first item in the tuple is a device
    array containing the vertices, the second item in the tuple is a device
    array containing the in degrees for the vertices.

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
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> (vertices, in_degrees) = pylibcugraph.in_degrees(
                                   resource_handle, G, None, False)

    """

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_degrees_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    assert_CAI_type(source_vertices, "source_vertices", True)

    cdef cugraph_type_erased_device_array_view_t* \
        source_vertices_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                source_vertices)

    error_code = cugraph_in_degrees(c_resource_handle_ptr,
                                    c_graph_ptr,
                                    source_vertices_ptr,
                                    do_expensive_check,
                                    &result_ptr,
                                    &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_in_degrees")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef cugraph_type_erased_device_array_view_t* vertices_ptr = \
        cugraph_degrees_result_get_vertices(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* in_degrees_ptr = \
        cugraph_degrees_result_get_in_degrees(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_in_degrees = copy_to_cupy_array(c_resource_handle_ptr, in_degrees_ptr)

    cugraph_degrees_result_free(result_ptr)

    return (cupy_vertices, cupy_in_degrees)

def out_degrees(ResourceHandle resource_handle,
                _GPUGraph graph,
                source_vertices,
                bool_t do_expensive_check):
    """
    Compute the out degrees for the nodes of the graph.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    source_vertices : cupy array
        The nodes for which we will compute degrees.

    do_expensive_check : bool_t
        A flag to run expensive checks for input arguments if True.

    Returns
    -------
    A tuple of device arrays, where the first item in the tuple is a device
    array containing the vertices, the second item in the tuple is a device
    array containing the out degrees for the vertices.

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
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> (vertices, out_degrees) = pylibcugraph.out_degrees(
                                    resource_handle, G, None, False)

    """

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_degrees_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    assert_CAI_type(source_vertices, "source_vertices", True)

    cdef cugraph_type_erased_device_array_view_t* \
        source_vertices_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                source_vertices)

    error_code = cugraph_out_degrees(c_resource_handle_ptr,
                                     c_graph_ptr,
                                     source_vertices_ptr,
                                     do_expensive_check,
                                     &result_ptr,
                                     &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_out_degrees")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef cugraph_type_erased_device_array_view_t* vertices_ptr = \
        cugraph_degrees_result_get_vertices(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* out_degrees_ptr = \
        cugraph_degrees_result_get_out_degrees(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_out_degrees = copy_to_cupy_array(c_resource_handle_ptr, out_degrees_ptr)

    cugraph_degrees_result_free(result_ptr)

    return (cupy_vertices, cupy_out_degrees)


def degrees(ResourceHandle resource_handle,
            _GPUGraph graph,
            source_vertices,
            bool_t do_expensive_check):
    """
    Compute the degrees for the nodes of the graph.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    source_vertices : cupy array
        The nodes for which we will compute degrees.

    do_expensive_check : bool_t
        A flag to run expensive checks for input arguments if True.

    Returns
    -------
    A tuple of device arrays, where the first item in the tuple is a device
    array containing the vertices, the second item in the tuple is a device
    array containing the in degrees for the vertices, the third item in the
    tuple is a device array containing the out degrees for the vertices.

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
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> (vertices, in_degrees, out_degrees) = pylibcugraph.degrees(
                                                resource_handle, G, None, False)

    """

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_degrees_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    assert_CAI_type(source_vertices, "source_vertices", True)

    cdef cugraph_type_erased_device_array_view_t* \
        source_vertices_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                source_vertices)

    error_code = cugraph_degrees(c_resource_handle_ptr,
                                 c_graph_ptr,
                                 source_vertices_ptr,
                                 do_expensive_check,
                                 &result_ptr,
                                 &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_degrees")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef cugraph_type_erased_device_array_view_t* vertices_ptr = \
        cugraph_degrees_result_get_vertices(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* in_degrees_ptr = \
        cugraph_degrees_result_get_in_degrees(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* out_degrees_ptr = \
        cugraph_degrees_result_get_out_degrees(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_in_degrees = copy_to_cupy_array(c_resource_handle_ptr, in_degrees_ptr)
    cupy_out_degrees = copy_to_cupy_array(c_resource_handle_ptr, out_degrees_ptr)

    cugraph_degrees_result_free(result_ptr)

    return (cupy_vertices, cupy_in_degrees, cupy_out_degrees)
