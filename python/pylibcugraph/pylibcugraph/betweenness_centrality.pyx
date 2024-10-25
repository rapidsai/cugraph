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
    cugraph_type_erased_device_array_view_free,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.centrality_algorithms cimport (
    cugraph_centrality_result_t,
    cugraph_betweenness_centrality,
    cugraph_centrality_result_get_vertices,
    cugraph_centrality_result_get_values,
    cugraph_centrality_result_free,
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
from pylibcugraph.select_random_vertices import (
    select_random_vertices
)


def betweenness_centrality(ResourceHandle resource_handle,
                           _GPUGraph graph,
                           k,
                           random_state,
                           bool_t normalized,
                           bool_t include_endpoints,
                           bool_t do_expensive_check):
    """
    Compute the betweenness centrality for all vertices of the graph G.
    Betweenness centrality is a measure of the number of shortest paths that
    pass through a vertex.  A vertex with a high betweenness centrality score
    has more paths passing through it and is therefore believed to be more
    important.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    k : int or device array type or None, optional (default=None)
        If k is not None, use k node samples to estimate betweenness.  Higher
        values give better approximation.  If k is a device array type,
        use the content of the list for estimation: the list should contain
        vertex identifiers. If k is None (the default), all the vertices are
        used to estimate betweenness.  Vertices obtained through sampling or
        defined as a list will be used as sources for traversals inside the
        algorithm.

    random_state : int, optional (default=None)
        if k is specified and k is an integer, use random_state to initialize the
        random number generator.
        Using None defaults to a hash of process id, time, and hostname
        If k is either None or list or cudf objects: random_state parameter is
        ignored.

    normalized : bool_t
        Normalization will ensure that values are in [0, 1].

    include_endpoints : bool_t
        If true, include the endpoints in the shortest path counts.

    do_expensive_check : bool_t
        A flag to run expensive checks for input arguments if True.

    Returns
    -------

    Examples
    --------

    """

    if isinstance(k, int):
        # randomly select vertices

        #'select_random_vertices' internally creates a
        # 'pylibcugraph.random.CuGraphRandomState'
        vertex_list = select_random_vertices(
            resource_handle, graph, random_state, k)
    else:
        vertex_list = k

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_centrality_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef cugraph_type_erased_device_array_view_t* \
        vertex_list_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                vertex_list)

    error_code = cugraph_betweenness_centrality(c_resource_handle_ptr,
                                                c_graph_ptr,
                                                vertex_list_view_ptr,
                                                normalized,
                                                include_endpoints,
                                                do_expensive_check,
                                                &result_ptr,
                                                &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_betweenness_centrality")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef cugraph_type_erased_device_array_view_t* vertices_ptr = \
        cugraph_centrality_result_get_vertices(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* values_ptr = \
        cugraph_centrality_result_get_values(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_values = copy_to_cupy_array(c_resource_handle_ptr, values_ptr)

    cugraph_centrality_result_free(result_ptr)
    cugraph_type_erased_device_array_view_free(vertex_list_view_ptr)

    return (cupy_vertices, cupy_values)
