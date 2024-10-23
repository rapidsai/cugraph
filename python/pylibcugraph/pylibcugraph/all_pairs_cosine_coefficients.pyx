# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
from libc.stdio cimport printf

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
from pylibcugraph._cugraph_c.graph_functions cimport (
    cugraph_vertex_pairs_t,
    cugraph_vertex_pairs_get_first,
    cugraph_vertex_pairs_get_second,
    cugraph_vertex_pairs_free,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.similarity_algorithms cimport (
    cugraph_all_pairs_cosine_similarity_coefficients,
    cugraph_similarity_result_t,
    cugraph_similarity_result_get_similarity,
    cugraph_similarity_result_get_vertex_pairs,
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
    create_cugraph_type_erased_device_array_view_from_py_obj,
    SIZE_MAX
)


def all_pairs_cosine_coefficients(ResourceHandle resource_handle,
        _GPUGraph graph,
        vertices,
        bool_t use_weight,
        topk,
        bool_t do_expensive_check):
    """
    Perform All-Pairs Cosine similarity computation.

    Note that Cosine similarity must run on a symmetric graph.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    vertices : cudf.Series or None
        Vertex list to compute all-pairs. If None, then compute based
            on all vertices in the graph.

    use_weight : bool, optional
        If set to True, then compute weighted cosine_coefficients(
            the input graph must be weighted in that case).
        Otherwise, compute non-weighted cosine_coefficients

    topk : size_t
        Specify the number of answers to return otherwise will return all values.


    do_expensive_check : bool
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    A tuple of device arrays containing the vertex pairs with
    their corresponding Cosine coefficient scores.

    Examples
    --------
    # FIXME: No example yet

    """

    if topk is None:
        topk = SIZE_MAX

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_similarity_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef cugraph_type_erased_device_array_view_t* \
        vertices_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                vertices)

    error_code = cugraph_all_pairs_cosine_similarity_coefficients(c_resource_handle_ptr,
                                              c_graph_ptr,
                                              vertices_view_ptr,
                                              use_weight,
                                              topk,
                                              do_expensive_check,
                                              &result_ptr,
                                              &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_all_pairs_cosine_similarity_coefficients")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef cugraph_type_erased_device_array_view_t* similarity_ptr = \
        cugraph_similarity_result_get_similarity(result_ptr)

    cupy_similarity = copy_to_cupy_array(c_resource_handle_ptr, similarity_ptr)

    cdef cugraph_vertex_pairs_t* vertex_pairs_ptr = \
        cugraph_similarity_result_get_vertex_pairs(result_ptr)

    cdef cugraph_type_erased_device_array_view_t* first_view_ptr = \
        cugraph_vertex_pairs_get_first(vertex_pairs_ptr)

    cupy_first = copy_to_cupy_array(c_resource_handle_ptr, first_view_ptr)

    cdef cugraph_type_erased_device_array_view_t* second_view_ptr = \
        cugraph_vertex_pairs_get_second(vertex_pairs_ptr)

    cupy_second = copy_to_cupy_array(c_resource_handle_ptr, second_view_ptr)

    # Free all pointers
    cugraph_similarity_result_free(result_ptr)
    cugraph_vertex_pairs_free(vertex_pairs_ptr)

    cugraph_type_erased_device_array_view_free(vertices_view_ptr)
    # No need to free 'first_view_ptr' and 'second_view_ptr' as their memory
    # are already deallocated when freeing 'result_ptr'

    return cupy_first, cupy_second, cupy_similarity
