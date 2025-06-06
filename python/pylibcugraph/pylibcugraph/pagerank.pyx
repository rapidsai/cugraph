# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

import cupy
import numpy

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
    cugraph_type_erased_device_array_view_free
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.centrality_algorithms cimport (
    cugraph_centrality_result_t,
    cugraph_pagerank_allow_nonconvergence,
    cugraph_centrality_result_converged,
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
    create_cugraph_type_erased_device_array_view_from_py_obj,
)
from pylibcugraph.exceptions import FailedToConvergeError


def pagerank(ResourceHandle resource_handle,
            _GPUGraph graph,
            precomputed_vertex_out_weight_vertices,
            precomputed_vertex_out_weight_sums,
            initial_guess_vertices,
            initial_guess_values,
            double alpha,
            double epsilon,
            size_t max_iterations,
            bool_t do_expensive_check,
            fail_on_nonconvergence=True):
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

    fail_on_nonconvergence : bool (default=True)
        If the solver does not reach convergence, raise an exception if
        fail_on_nonconvergence is True. If fail_on_nonconvergence is False,
        the return value is a tuple of (pagerank, converged) where pagerank is
        a cudf.DataFrame as described below, and converged is a boolean
        indicating if the solver converged (True) or not (False).

    Returns
    -------
    The return value varies based on the value of the fail_on_nonconvergence
    paramter.  If fail_on_nonconvergence is True:

       A tuple of device arrays, where the first item in the tuple is a device
       array containing the vertex identifiers, and the second item is a device
       array containing the pagerank values for the corresponding vertices. For
       example, the vertex identifier at the ith element of the vertex array
       has the pagerank value of the ith element in the pagerank array.

    If fail_on_nonconvergence is False:

       A three-tuple where the first two items are the device arrays described
       above, and the third is a bool indicating if the solver converged (True)
       or not (False).

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
    >>> (vertices, pageranks) = pylibcugraph.pagerank(
    ...     resource_handle, G, None, None, None, None, alpha=0.85,
    ...     epsilon=1.0e-6, max_iterations=500, do_expensive_check=False)
    >>> vertices
    array([0, 1, 2, 3], dtype=int32)
    >>> pageranks
    array([0.11615585, 0.21488841, 0.2988108 , 0.3701449 ], dtype=float32)
    """

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

    cdef cugraph_centrality_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr
    cdef bool_t converged
    cdef cugraph_type_erased_device_array_view_t* vertices_ptr
    cdef cugraph_type_erased_device_array_view_t* pageranks_ptr

    error_code = cugraph_pagerank_allow_nonconvergence(
        c_resource_handle_ptr,
        c_graph_ptr,
        precomputed_vertex_out_weight_vertices_view_ptr,
        precomputed_vertex_out_weight_sums_view_ptr,
        initial_guess_vertices_view_ptr,
        initial_guess_values_view_ptr,
        alpha,
        epsilon,
        max_iterations,
        do_expensive_check,
        &result_ptr,
        &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_pagerank_allow_nonconvergence")

    converged = cugraph_centrality_result_converged(result_ptr)

    # Only extract results if necessary
    if (fail_on_nonconvergence is False) or (converged is True):
        # Extract individual device array pointers from result and copy to cupy
        # arrays for returning.
        vertices_ptr = cugraph_centrality_result_get_vertices(result_ptr)
        pageranks_ptr = cugraph_centrality_result_get_values(result_ptr)
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

    if fail_on_nonconvergence is False:
        return (cupy_vertices, cupy_pageranks, bool(converged))
    else:
        if converged is True:
            return (cupy_vertices, cupy_pageranks)
        else:
            raise FailedToConvergeError
