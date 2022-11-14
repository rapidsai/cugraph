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
    cugraph_resource_handle_t,
    bool_t,
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
from pylibcugraph._cugraph_c.graph_functions cimport (
    cugraph_two_hop_neighbors,
    cugraph_vertex_pairs_t,
    cugraph_vertex_pairs_get_first,
    cugraph_vertex_pairs_get_second,
    cugraph_vertex_pairs_free,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
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
    create_cugraph_type_erased_device_array_view_from_py_obj
)


def get_two_hop_neighbors(ResourceHandle resource_handle,
                          _GPUGraph graph,
                          start_vertices,
                          bool_t do_expensive_check):
    """
        Compute vertex pairs that are two hops apart. The resulting pairs are
        sorted before returning.

        Parameters
        ----------
        resource_handle : ResourceHandle
            Handle to the underlying device resources needed for referencing data
            and running algorithms.

        graph : SGGraph or MGGraph
            The input graph, for either Single or Multi-GPU operations.
        
        start_vertices : Optional array of starting vertices
                         If None use all, if specified compute two-hop
                         neighbors for these starting vertices

        Returns
        -------
        return a cupy arrays of 'first' and 'second' or a 'cugraph_vertex_pairs_t'
        which can be directly passed to the similarity algorithm?
    """

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_vertex_pairs_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef cugraph_type_erased_device_array_view_t* start_vertices_ptr

    cdef cugraph_type_erased_device_array_view_t* \
        start_vertices_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                start_vertices)

    error_code = cugraph_two_hop_neighbors(c_resource_handle_ptr,
                                           c_graph_ptr,
                                           start_vertices_view_ptr,
                                           do_expensive_check,
                                           &result_ptr,
                                           &error_ptr)
    assert_success(error_code, error_ptr, "two_hop_neighbors")

    cdef cugraph_type_erased_device_array_view_t* first_ptr = \
        cugraph_vertex_pairs_get_first(result_ptr)
    
    cdef cugraph_type_erased_device_array_view_t* second_ptr = \
        cugraph_vertex_pairs_get_second(result_ptr)
    
    cupy_first = copy_to_cupy_array(c_resource_handle_ptr, first_ptr)
    cupy_second = copy_to_cupy_array(c_resource_handle_ptr, second_ptr)

    # Free all pointers
    cugraph_vertex_pairs_free(result_ptr)
    if start_vertices is not None:
        cugraph_type_erased_device_array_view_free(start_vertices_view_ptr)    

    return cupy_first, cupy_second
