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
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.algorithms cimport (
    cugraph_sssp,
    cugraph_paths_result_t,
    cugraph_paths_result_get_vertices,
    cugraph_paths_result_get_distances,
    cugraph_paths_result_get_predecessors,
    cugraph_paths_result_free,
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


def sssp(ResourceHandle resource_handle,
        _GPUGraph graph,
        size_t source,
        double cutoff,
        bool_t compute_predecessors,
        bool_t do_expensive_check):
    """
    Compute the distance and predecessors for shortest paths from the specified
    source to all the vertices in the graph. The returned distances array will
    contain the distance from the source to each vertex in the returned vertex
    array at the same index. The returned predecessors array will contain the
    previous vertex in the shortest path for each vertex in the vertex array at
    the same index. Vertices that are unreachable will have a distance of
    infinity denoted by the maximum value of the data type and the predecessor
    set as -1. The source vertex predecessor will be set to -1. Graphs with
    negative weight cycles are not supported.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    source :
        The vertex identifier of the source vertex.

    cutoff :
        Maximum edge weight sum to consider.

    compute_predecessors : bool
       This parameter must be set to True for this release.

    do_expensive_check : bool
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    A 3-tuple, where the first item in the tuple is a device array containing
    the vertex identifiers, the second item is a device array containing the
    distance for each vertex from the source vertex, and the third item is a
    device array containing the vertex identifier of the preceding vertex in the
    path for that vertex. For example, the vertex identifier at the ith element
    of the vertex array has a distance from the source vertex of the ith element
    in the distance array, and the preceding vertex in the path is the ith
    element in the predecessor array.

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
    ...     store_transposed=False, renumber=False, do_expensive_check=False)
    >>> (vertices, distances, predecessors) = pylibcugraph.sssp(
    ...     resource_handle, G, source=1, cutoff=999,
    ...     compute_predecessors=True, do_expensive_check=False)
    >>> vertices
    array([0, 1, 2, 3], dtype=int32)
    >>> distances
    array([3.4028235e+38, 0.0000000e+00, 1.0000000e+00, 2.0000000e+00],
          dtype=float32)
    >>> predecessors
    array([-1, -1,  1,  2], dtype=int32)
    """

    # FIXME: import these modules here for now until a better pattern can be
    # used for optional imports (perhaps 'import_optional()' from cugraph), or
    # these are made hard dependencies.
    try:
        import cupy
    except ModuleNotFoundError:
        raise RuntimeError("sssp requires the cupy package, which could not "
                           "be imported")
    try:
        import numpy
    except ModuleNotFoundError:
        raise RuntimeError("sssp requires the numpy package, which could not "
                           "be imported")

    if compute_predecessors is False:
        raise ValueError("compute_predecessors must be True for the current "
                         "release.")

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_paths_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    error_code = cugraph_sssp(c_resource_handle_ptr,
                              c_graph_ptr,
                              source,
                              cutoff,
                              compute_predecessors,
                              do_expensive_check,
                              &result_ptr,
                              &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_sssp")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef cugraph_type_erased_device_array_view_t* vertices_ptr = \
        cugraph_paths_result_get_vertices(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* distances_ptr = \
        cugraph_paths_result_get_distances(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* predecessors_ptr = \
        cugraph_paths_result_get_predecessors(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_distances = copy_to_cupy_array(c_resource_handle_ptr, distances_ptr)
    cupy_predecessors = copy_to_cupy_array(c_resource_handle_ptr,
                                           predecessors_ptr)

    cugraph_paths_result_free(result_ptr)

    return (cupy_vertices, cupy_distances, cupy_predecessors)
