# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
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
    cugraph_type_erased_device_array_t,
    cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_device_array_view
)
from pylibcugraph._cugraph_c.graph_functions cimport (
    cugraph_has_vertex
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)
from pylibcugraph.graphs cimport (
    _GPUGraph,
)
from pylibcugraph.utils cimport (
    assert_success,
    assert_CAI_type,
    copy_to_cupy_array,
    create_cugraph_type_erased_device_array_view_from_py_obj
)


def has_vertex(ResourceHandle resource_handle,
               _GPUGraph graph,
               vertices,
               bool_t do_expensive_check):
    """
        Verify if vertices exists in the graph

        Parameters
        ----------
        resource_handle : ResourceHandle
            Handle to the underlying device resources needed for referencing data
            and running algorithms.

        graph : SGGraph or MGGraph
            The input graph, for either Single or Multi-GPU operations.

        vertices : device array type
                 array of vertices to be queried

        Returns
        -------
        Return a device array of bool where 'True' is indicative of a vertex existance
        and 'False' otherwise.

        Examples
        --------
        >>> import pylibcugraph, cupy, numpy
        >>> srcs = cupy.asarray([0, 1, 1, 2, 2, 2, 3, 3, 4], dtype=numpy.int32)
        >>> dsts = cupy.asarray([1, 3, 4, 0, 1, 3, 4, 5, 5], dtype=numpy.int32)
        >>> weights = cupy.asarray(
        ...     [0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2, 6.1], dtype=numpy.float32)
        >>> source_vertices = cupy.asarray([0, 1], dtype=numpy.int32)
        >>> resource_handle = pylibcugraph.ResourceHandle()
        >>> graph_props = pylibcugraph.GraphProperties(
        ...     is_symmetric=False, is_multigraph=False)
        >>> G = pylibcugraph.SGGraph(
        ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
        ...     store_transposed=False, renumber=False, do_expensive_check=False)
        >>> vertices = cupy.asarray([0, 1, 2, 6], dtype=numpy.int32)
        >>> result = pylibcugraph.has_vertex(resource_handle, G, vertices False)
        >>> result
        [ True  True  True False]
        """

    assert_CAI_type(vertices, "vertices")

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_type_erased_device_array_view_t* \
        vertices_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                vertices)

    cdef cugraph_type_erased_device_array_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    error_code = cugraph_has_vertex(c_resource_handle_ptr,
                                    c_graph_ptr,
                                    vertices_view_ptr,
                                    do_expensive_check,
                                    &result_ptr,
                                    &error_ptr)
    assert_success(error_code, error_ptr, "has_vertex")

    cdef cugraph_type_erased_device_array_view_t* \
        result_view_ptr = \
            cugraph_type_erased_device_array_view(
                result_ptr)

    cupy_has_vertex = copy_to_cupy_array(c_resource_handle_ptr, result_view_ptr)

    return cupy_has_vertex
