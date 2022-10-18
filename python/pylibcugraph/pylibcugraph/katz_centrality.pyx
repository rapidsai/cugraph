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
from pylibcugraph._cugraph_c.centrality_algorithms cimport (
    cugraph_centrality_result_t,
    cugraph_katz_centrality,
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
    get_c_type_from_numpy_type,
)


def katz_centrality(ResourceHandle resource_handle,
                    _GPUGraph graph,
                    betas,
                    double alpha,
                    double beta,
                    double epsilon,
                    size_t max_iterations,
                    bool_t do_expensive_check):
    """
    Compute the Katz centrality for the nodes of the graph. This implementation
    is based on a relaxed version of Katz defined by Foster with a reduced
    computational complexity of O(n+m)

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    betas : device array type
        Device array containing the values to be added to each vertex's new
        Katz Centrality score in every iteration. If set to None then beta is
        used for all vertices.

    alpha : double
        The attenuation factor, should be smaller than the inverse of the
        maximum eigenvalue of the graph

    beta : double
        Constant value to be added to each vertex's new Katz Centrality score
        in every iteration. Relevant only when betas is None

    epsilon : double
        Error tolerance to check convergence

    max_iterations: size_t
        Maximum number of Katz Centrality iterations

    do_expensive_check : bool_t
        A flag to run expensive checks for input arguments if True.
    
    Returns
    -------

    Examples
    --------

    """

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_centrality_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef uintptr_t cai_betas_ptr 
    cdef cugraph_type_erased_device_array_view_t* betas_ptr
    
    if betas is not None:
        cai_betas_ptr = betas.__cuda_array_interface__["data"][0]
        betas_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_betas_ptr,
                len(betas),
                get_c_type_from_numpy_type(betas.dtype))
    else:
        betas_ptr = NULL

    error_code = cugraph_katz_centrality(c_resource_handle_ptr,
                                         c_graph_ptr,
                                         betas_ptr,
                                         alpha,
                                         beta,
                                         epsilon,
                                         max_iterations,
                                         do_expensive_check,
                                         &result_ptr,
                                         &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_katz_centrality")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef cugraph_type_erased_device_array_view_t* vertices_ptr = \
        cugraph_centrality_result_get_vertices(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* values_ptr = \
        cugraph_centrality_result_get_values(result_ptr)
    
    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_values = copy_to_cupy_array(c_resource_handle_ptr, values_ptr)

    cugraph_centrality_result_free(result_ptr)
    cugraph_type_erased_device_array_view_free(betas_ptr)

    return (cupy_vertices, cupy_values)
