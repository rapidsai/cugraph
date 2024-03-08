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
    copy_to_cupy_array,
)
from pylibcugraph._cugraph_c.random cimport (
    cugraph_rng_state_t
)
from pylibcugraph.random cimport (
    CuGraphRandomState
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_t,
    cugraph_type_erased_device_array_view
)
from pylibcugraph._cugraph_c.sampling_algorithms cimport (
    cugraph_select_random_vertices
)


def select_random_vertices(ResourceHandle resource_handle,
                           _GPUGraph graph,
                           random_state,
                           size_t num_vertices,
                           ):
    """
    Select random vertices from the graph

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    random_state : int , optional
        Random state to use when generating samples. Optional argument,
        defaults to a hash of process id, time, and hostname.
        (See pylibcugraph.random.CuGraphRandomState)

    num_vertices : size_t , optional
        Number of vertices to sample. Optional argument, defaults to the
        total number of vertices.

    Returns
    -------
    return random vertices from the graph
    """

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_type_erased_device_array_t* vertices_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cg_rng_state = CuGraphRandomState(resource_handle, random_state)

    cdef cugraph_rng_state_t* rng_state_ptr = \
        cg_rng_state.rng_state_ptr

    error_code = cugraph_select_random_vertices(c_resource_handle_ptr,
                                                c_graph_ptr,
                                                rng_state_ptr,
                                                num_vertices,
                                                &vertices_ptr,
                                                &error_ptr)
    assert_success(error_code, error_ptr, "select_random_vertices")

    cdef cugraph_type_erased_device_array_view_t* \
        vertices_view_ptr = \
            cugraph_type_erased_device_array_view(
                vertices_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_view_ptr)

    return cupy_vertices
