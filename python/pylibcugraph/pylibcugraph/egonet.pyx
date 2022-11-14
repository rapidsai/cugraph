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

from pylibcugraph._cugraph_c.community_algorithms cimport (
    cugraph_extract_ego,
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
    copy_to_cupy_array_weights_offsets,
    create_cugraph_type_erased_device_array_view_from_py_obj,
)


def ego_graph(ResourceHandle resource_handle,
              _GPUGraph graph,
              source_vertices,
              size_t radius,
              bool_t do_expensive_check):
    """
    Compute the induced subgraph of neighbors centered at nodes
    source_vertices, within a given radius.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph.
    
    source_vertices : cudf.Series
        The centered nodes from which the induced subgraph will be extracted

    radius: size_t
        The number of hops to go out from each source vertex


    do_expensive_check : bool_t
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    A tuple of device arrays containing the sources, destinations, edge_weights
    and the subgraph_offsets(if there are more than one seeds)

    Examples
    --------
    >>> import pylibcugraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 0], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> source_vertices = cupy.asarray([1, 2], dtype=numpy.int32)
    >>> resource_handle = pylibcugraph.ResourceHandle()
    >>> graph_props = pylibcugraph.GraphProperties(
    ...     is_symmetric=True, is_multigraph=False)
    >>> G = pylibcugraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> (sources, destinations, edge_weights, subgraph_offsets) =
    ...     pylibcugraph.ego_graph(resource_handle, G, source_vertices, 1, False)
    # FIXME: update results
    >>> sources
    [x, x, x]
    >>> destinations
    [x, x, x]
    >>> edge_weights
    [x, x, x]
    >>> subgraph_offsets
    [x, x, x]

    """
    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr
    cdef cugraph_induced_subgraph_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef cugraph_type_erased_device_array_view_t* \
        source_vertices_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                source_vertices)

    error_code = cugraph_extract_ego(c_resource_handle_ptr,
                                     c_graph_ptr,
                                     source_vertices_view_ptr,
                                     radius,
                                     do_expensive_check,
                                     &result_ptr,
                                     &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_extract_ego")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef cugraph_type_erased_device_array_view_t* sources_ptr = \
        cugraph_induced_subgraph_get_sources(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* destinations_ptr = \
        cugraph_induced_subgraph_get_destinations(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* edge_weights_ptr = \
        cugraph_induced_subgraph_get_edge_weights(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* subgraph_offsets_ptr

    cupy_sources = copy_to_cupy_array(c_resource_handle_ptr, sources_ptr)
    cupy_destinations = copy_to_cupy_array(
        c_resource_handle_ptr, destinations_ptr)
    cupy_edge_weights = copy_to_cupy_array_weights_offsets(
        c_resource_handle_ptr, edge_weights_ptr, "weights")
    
    subgraph_offsets_ptr = cugraph_induced_subgraph_get_subgraph_offsets(
            result_ptr)
    
    # FIXME: The C enum value for offsets is 4. For the rest is 0
    # which is also a problem for weights
    cupy_subgraph_offsets = copy_to_cupy_array_weights_offsets(
        c_resource_handle_ptr, subgraph_offsets_ptr, "offsets")
    
    cugraph_induced_subgraph_result_free(result_ptr)

    return (cupy_sources, cupy_destinations,
                cupy_edge_weights, cupy_subgraph_offsets)
