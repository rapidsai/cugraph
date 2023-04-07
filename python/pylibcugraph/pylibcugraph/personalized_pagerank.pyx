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
    cugraph_type_erased_device_array_view_free
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.centrality_algorithms cimport (
    cugraph_centrality_result_t,
    cugraph_personalized_pagerank,
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
    assert_CAI_type,
    copy_to_cupy_array,
    get_c_type_from_numpy_type,
    create_cugraph_type_erased_device_array_view_from_py_obj,
)


def personalized_pagerank(ResourceHandle resource_handle,
                          _GPUGraph graph,
                          precomputed_vertex_out_weight_vertices,
                          precomputed_vertex_out_weight_sums,
                          initial_guess_vertices,
                          initial_guess_values,
                          personalization_vertices,
                          personalization_values,
                          double alpha,
                          double epsilon,
                          size_t max_iterations,
                          bool_t do_expensive_check):
    """
    Find the PageRank score for every vertex in a graph by computing an
    approximation of the Pagerank eigenvector using the power method. The
    number of iterations depends on the properties of the network itself; it
    increases when the tolerance descreases and/or alpha increases toward the
    limiting value of 1.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    precomputed_vertex_out_weight_vertices: device array type
        Subset of vertices of graph for precomputed_vertex_out_weight
        (a performance optimization)

    precomputed_vertex_out_weight_sums : device array type
        Corresponding precomputed sum of outgoing vertices weight
        (a performance optimization)
    
    initial_guess_vertices : device array type
        Subset of vertices of graph for initial guess for pagerank values
        (a performance optimization)
    
    initial_guess_values : device array type
        Pagerank values for vertices
        (a performance optimization)
    
    personalization_vertices : device array type
        Subset of vertices of graph for personalization
        (a performance optimization)
    
    personalization_values : device array type
        Personalization values for vertices
        (a performance optimization)

    alpha : double
        The damping factor alpha represents the probability to follow an
        outgoing edge, standard value is 0.85.
        Thus, 1.0-alpha is the probability to “teleport” to a random vertex.
        Alpha should be greater than 0.0 and strictly lower than 1.0.

    epsilon : double
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.
        The lower the tolerance the better the approximation. If this value is
        0.0f, cuGraph will use the default value which is 1.0E-5.
        Setting too small a tolerance can lead to non-convergence due to
        numerical roundoff. Usually values between 0.01 and 0.00001 are
        acceptable.

    max_iterations : size_t
        The maximum number of iterations before an answer is returned. This can
        be used to limit the execution time and do an early exit before the
        solver reaches the convergence tolerance.
        If this value is lower or equal to 0 cuGraph will use the default
        value, which is 100.

    do_expensive_check : bool_t
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    A tuple of device arrays, where the first item in the tuple is a device
    array containing the vertex identifiers, and the second item is a device
    array containing the pagerank values for the corresponding vertices. For
    example, the vertex identifier at the ith element of the vertex array has
    the pagerank value of the ith element in the pagerank array.

    Examples
    --------
    >>> import pylibcugraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 3], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> personalization_vertices = cupy.asarray([0, 2], dtype=numpy.int32)
    >>> personalization_values = cupy.asarray(
    ...     [0.008309, 0.991691], dtype=numpy.float32)
    >>> resource_handle = pylibcugraph.ResourceHandle()
    >>> graph_props = pylibcugraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibcugraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> (vertices, pageranks) = pylibcugraph.personalized_pagerank(
    ...     resource_handle, G, None, None, None, None, alpha=0.85,
    ...     personalization_vertices=personalization_vertices,
    ...     personalization_values=personalization_values, epsilon=1.0e-6,
    ...     max_iterations=500,
    ...     do_expensive_check=False)
    >>> vertices
    array([0, 1, 2, 3], dtype=int32)
    >>> pageranks
    array([0.00446455, 0.00379487, 0.53607565, 0.45566472 ], dtype=float32)
    """

    # FIXME: import these modules here for now until a better pattern can be
    # used for optional imports (perhaps 'import_optional()' from cugraph), or
    # these are made hard dependencies.
    try:
        import cupy
    except ModuleNotFoundError:
        raise RuntimeError("pagerank requires the cupy package, which could "
                           "not be imported")
    try:
        import numpy
    except ModuleNotFoundError:
        raise RuntimeError("pagerank requires the numpy package, which could "
                           "not be imported")

    cdef cugraph_type_erased_device_array_view_t* \
        initial_guess_vertices_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                initial_guess_vertices)

    cdef cugraph_type_erased_device_array_view_t* \
        initial_guess_values_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                initial_guess_values)

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_type_erased_device_array_view_t* \
        precomputed_vertex_out_weight_vertices_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                precomputed_vertex_out_weight_vertices)

    # FIXME: assert that precomputed_vertex_out_weight_sums
    # type == weight type
    cdef cugraph_type_erased_device_array_view_t* \
        precomputed_vertex_out_weight_sums_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                precomputed_vertex_out_weight_sums)
    
    cdef cugraph_type_erased_device_array_view_t* \
        personalization_vertices_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                personalization_vertices)
    
    cdef cugraph_type_erased_device_array_view_t* \
        personalization_values_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                personalization_values)

    cdef cugraph_centrality_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    error_code = cugraph_personalized_pagerank(c_resource_handle_ptr,
                                               c_graph_ptr,
                                               precomputed_vertex_out_weight_vertices_view_ptr,
                                               precomputed_vertex_out_weight_sums_view_ptr,
                                               initial_guess_vertices_view_ptr,
                                               initial_guess_values_view_ptr,
                                               personalization_vertices_view_ptr,
                                               personalization_values_view_ptr,
                                               alpha,
                                               epsilon,
                                               max_iterations,
                                               do_expensive_check,
                                               &result_ptr,
                                               &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_personalized_pagerank")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef cugraph_type_erased_device_array_view_t* vertices_ptr = \
        cugraph_centrality_result_get_vertices(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* pageranks_ptr = \
        cugraph_centrality_result_get_values(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_pageranks = copy_to_cupy_array(c_resource_handle_ptr, pageranks_ptr)

    # Free all pointers
    cugraph_centrality_result_free(result_ptr)
    if initial_guess_vertices is not None:
        cugraph_type_erased_device_array_view_free(initial_guess_vertices_view_ptr)
    if initial_guess_values is not None:
        cugraph_type_erased_device_array_view_free(initial_guess_values_view_ptr)
    if precomputed_vertex_out_weight_vertices is not None:
        cugraph_type_erased_device_array_view_free(precomputed_vertex_out_weight_vertices_view_ptr)
    if precomputed_vertex_out_weight_sums is not None:
        cugraph_type_erased_device_array_view_free(precomputed_vertex_out_weight_sums_view_ptr)
    if personalization_vertices is not None:
        cugraph_type_erased_device_array_view_free(personalization_vertices_view_ptr)
    if personalization_values is not None:
        cugraph_type_erased_device_array_view_free(personalization_values_view_ptr)

    return (cupy_vertices, cupy_pageranks)
