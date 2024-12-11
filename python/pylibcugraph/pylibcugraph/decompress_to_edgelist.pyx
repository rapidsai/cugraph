# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
    cugraph_edgelist_t,
    cugraph_decompress_to_edgelist,
    cugraph_edgelist_get_sources,
    cugraph_edgelist_get_destinations,
    cugraph_edgelist_get_edge_weights,
    cugraph_edgelist_get_edge_ids,
    cugraph_edgelist_get_edge_type_ids,
    cugraph_edgelist_get_edge_offsets,
    cugraph_edgelist_free,
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
    create_cugraph_type_erased_device_array_view_from_py_obj,
)


def decompress_to_edgelist(ResourceHandle resource_handle,
                           _GPUGraph graph,
                           bool_t do_expensive_check):
    """
    Extract a the edgelist from a graph

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph.

    do_expensive_check : bool_t
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    A tuple of device arrays containing the sources, destinations and if applicable
    edge_weights, edge_ids and/or edge_type_ids.

    Examples
    --------
    >>> import pylibcugraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 1, 2, 2, 2, 3, 4], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 3, 4, 0, 1, 3, 5, 5], dtype=numpy.int32)
    >>> weights = cupy.asarray(
    ...     [0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2], dtype=numpy.float32)
    >>> resource_handle = pylibcugraph.ResourceHandle()
    >>> graph_props = pylibcugraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibcugraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=False, renumber=False, do_expensive_check=False)
    >>> (sources, destinations, edge_weights, _, _) =
    ...     pylibcugraph.decompress_to_edgelist(
    ...         resource_handle, G, False)
    >>> sources
    [0, 1, 1, 2, 2, 2, 3, 4]
    >>> destinations
    [1, 3, 4, 0, 1, 3, 5, 5]
    >>> edge_weights
    [0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2]
    """

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr
    cdef cugraph_edgelist_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    error_code = cugraph_decompress_to_edgelist(c_resource_handle_ptr,
                                                c_graph_ptr,
                                                do_expensive_check,
                                                &result_ptr,
                                                &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_decompress_to_edgelist")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef cugraph_type_erased_device_array_view_t* sources_ptr = \
        cugraph_edgelist_get_sources(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* destinations_ptr = \
        cugraph_edgelist_get_destinations(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* edge_weights_ptr = \
        cugraph_edgelist_get_edge_weights(result_ptr)

    cdef cugraph_type_erased_device_array_view_t* edge_ids_ptr = \
        cugraph_edgelist_get_edge_ids(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* edge_type_ids_ptr = \
        cugraph_edgelist_get_edge_type_ids(result_ptr)


    """
    cdef cugraph_type_erased_device_array_view_t* subgraph_offsets_ptr = \
        cugraph_edgelist_get_edge_offsets(result_ptr)
    """

    # FIXME: Get ownership of the result data instead of performing a copy
    # for perfomance improvement
    cupy_edge_weights = None
    cupy_edge_ids = None
    cupy_edge_type_ids = None
    cupy_sources = copy_to_cupy_array(
        c_resource_handle_ptr, sources_ptr)
    cupy_destinations = copy_to_cupy_array(
        c_resource_handle_ptr, destinations_ptr)
    if edge_weights_ptr != NULL:
        cupy_edge_weights = copy_to_cupy_array(
            c_resource_handle_ptr, edge_weights_ptr)
    if edge_ids_ptr != NULL:
        cupy_edge_ids = copy_to_cupy_array(
            c_resource_handle_ptr, edge_ids_ptr)
    if edge_type_ids_ptr != NULL:
        cupy_edge_type_ids = copy_to_cupy_array(
            c_resource_handle_ptr, edge_type_ids_ptr)

    """
    cupy_subgraph_offsets = copy_to_cupy_array(
        c_resource_handle_ptr, subgraph_offsets_ptr)
    """

    # Free pointer
    cugraph_edgelist_free(result_ptr)

    return (cupy_sources, cupy_destinations,
                cupy_edge_weights, cupy_edge_ids, cupy_edge_type_ids)
