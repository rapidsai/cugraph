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
    cugraph_type_erased_device_array_view_t,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.graph_functions cimport (
    cugraph_induced_subgraph_result_t,
    cugraph_induced_subgraph_get_sources,
    cugraph_induced_subgraph_get_destinations,
    cugraph_induced_subgraph_get_edge_weights,
    cugraph_induced_subgraph_get_subgraph_offsets,
    cugraph_induced_subgraph_result_free,
)
from pylibcugraph._cugraph_c.tree_algorithms cimport (
    cugraph_minimum_spanning_tree,
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
)


def minimum_spanning_tree(ResourceHandle resource_handle,
                          _GPUGraph graph,
                          do_expensive_check=False):
    """
    Extract a minimum spanning tree (MST) or forest (MSF) on an undirected
    graph

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph.

    do_expensive_check : bool (default=True)
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    A tuple of device arrays containing the sources, destinations,
    edge_weights and edge_offsets.

    Examples
    --------
    >>> import pylibcugraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4,
    ...     0, 1, 3, 5, 5], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1,
    ...     2, 2, 2, 3, 4], dtype=numpy.int32)
    >>> weights = cupy.asarray(
    ...     [0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2, 0.1, 2.1,
    ...     1.1, 5.1, 3.1, 4.1, 7.2, 3.2]
    ...     ,dtype=numpy.float32)
    >>> resource_handle = pylibcugraph.ResourceHandle()
    >>> graph_props = pylibcugraph.GraphProperties(
    ...     is_symmetric=True, is_multigraph=False)
    >>> G = pylibcugraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=False, renumber=False, do_expensive_check=False)
    >>> (sources, destinations, edge_weights, subgraph_offsets) =
    ...     pylibcugraph.minimum_spanning_tree(resource_handle, G, False)
    >>> sources
    array([0, 1, 2, 3, 4, 5, 1, 1, 1, 4], dtype=int32)
    >>> destinations
    array([1, 0, 1, 1, 1, 4, 2, 3, 4, 5], dtype=int32)
    >>> edge_weights
    array([0.1, 0.1, 3.1, 2.1, 1.1, 3.2, 3.1, 2.1, 1.1, 3.2], dtype=float32)
    >>> subgraph_offsets
    array([0, 10])

    """
    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr
    cdef cugraph_induced_subgraph_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    error_code = cugraph_minimum_spanning_tree(c_resource_handle_ptr,
                                               c_graph_ptr,
                                               do_expensive_check,
                                               &result_ptr,
                                               &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_minimum_spanning_tree")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef cugraph_type_erased_device_array_view_t* sources_ptr = \
        cugraph_induced_subgraph_get_sources(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* destinations_ptr = \
        cugraph_induced_subgraph_get_destinations(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* edge_weights_ptr = \
        cugraph_induced_subgraph_get_edge_weights(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* subgraph_offsets_ptr = \
        cugraph_induced_subgraph_get_subgraph_offsets(result_ptr)


    # FIXME: Get ownership of the result data instead of performing a copy
    # for perfomance improvement
    cupy_sources = copy_to_cupy_array(
        c_resource_handle_ptr, sources_ptr)

    cupy_destinations = copy_to_cupy_array(
        c_resource_handle_ptr, destinations_ptr)

    if edge_weights_ptr is not NULL:
        cupy_edge_weights = copy_to_cupy_array(
            c_resource_handle_ptr, edge_weights_ptr)
    else:
        cupy_edge_weights = None

    cupy_subgraph_offsets = copy_to_cupy_array(
        c_resource_handle_ptr, subgraph_offsets_ptr)

    # Free pointer
    cugraph_induced_subgraph_result_free(result_ptr)

    return (cupy_sources, cupy_destinations, cupy_edge_weights, cupy_subgraph_offsets)
