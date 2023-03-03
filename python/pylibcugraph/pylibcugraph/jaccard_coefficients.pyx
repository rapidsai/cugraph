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
from pylibcugraph._cugraph_c.graph_functions cimport (
    cugraph_vertex_pairs_t,
    cugraph_vertex_pairs_get_first,
    cugraph_vertex_pairs_get_second,
    cugraph_vertex_pairs_free,
    cugraph_create_vertex_pairs
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.similarity_algorithms cimport (
    cugraph_jaccard_coefficients,
    cugraph_similarity_result_t,
    cugraph_similarity_result_get_similarity,
    cugraph_similarity_result_free
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
    create_cugraph_type_erased_device_array_view_from_py_obj
)


def EXPERIMENTAL__jaccard_coefficients(ResourceHandle resource_handle,
        _GPUGraph graph,
        first,
        second,
        bool_t use_weight,
        bool_t do_expensive_check):
    """
    Compute the Jaccard coefficients for the specified vertex_pairs.
    
    Note that Jaccard similarity must run on a symmetric graph.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.
    
    first :
        Source of the vertex pair.
    
    second :
        Destination of the vertex pair.
    
    use_weight : bool, optional (default=False)
        Currently not supported

    do_expensive_check : bool
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    A tuple of device arrays containing the vertex pairs with
    their corresponding Jaccard coefficient scores.

    Examples
    --------
    # FIXME: No example yet

    """

    cdef cugraph_vertex_pairs_t* vertex_pairs_ptr

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_similarity_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    # 'first' is a required parameter
    cdef cugraph_type_erased_device_array_view_t* \
        first_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                first)

    # 'second' is a required parameter
    cdef cugraph_type_erased_device_array_view_t* \
        second_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                second)

    error_code = cugraph_create_vertex_pairs(c_resource_handle_ptr,
                                             c_graph_ptr,
                                             first_view_ptr,
                                             second_view_ptr,
                                             &vertex_pairs_ptr,
                                             &error_ptr)
    assert_success(error_code, error_ptr, "vertex_pairs")

    error_code = cugraph_jaccard_coefficients(c_resource_handle_ptr,
                                              c_graph_ptr,
                                              vertex_pairs_ptr,
                                              use_weight,
                                              do_expensive_check,
                                              &result_ptr,
                                              &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_jaccard_coefficients")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef cugraph_type_erased_device_array_view_t* similarity_ptr = \
        cugraph_similarity_result_get_similarity(result_ptr)

    cupy_similarity = copy_to_cupy_array(c_resource_handle_ptr, similarity_ptr)

    cdef cugraph_type_erased_device_array_view_t* first_ptr = \
        cugraph_vertex_pairs_get_first(vertex_pairs_ptr)

    cupy_first = copy_to_cupy_array(c_resource_handle_ptr, first_ptr)

    cdef cugraph_type_erased_device_array_view_t* second_ptr = \
        cugraph_vertex_pairs_get_second(vertex_pairs_ptr)

    cupy_second = copy_to_cupy_array(c_resource_handle_ptr, second_ptr)

    # Free all pointers
    cugraph_similarity_result_free(result_ptr)
    cugraph_vertex_pairs_free(vertex_pairs_ptr)

    cugraph_type_erased_device_array_view_free(first_view_ptr)
    cugraph_type_erased_device_array_view_free(second_view_ptr)

    return cupy_first, cupy_second, cupy_similarity
