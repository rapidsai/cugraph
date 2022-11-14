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
    cugraph_type_erased_host_array_view_t,
    cugraph_type_erased_host_array_view_create,
    cugraph_type_erased_host_array_view_free,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.algorithms cimport (
    cugraph_uniform_random_walks,
    cugraph_random_walk_result_t,
    cugraph_random_walk_result_get_paths,
    cugraph_random_walk_result_get_weights,
    cugraph_random_walk_result_get_path_sizes,
    cugraph_random_walk_result_get_max_path_length,
    cugraph_random_walk_result_free,
)
from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)
from pylibcugraph.graphs cimport (
    _GPUGraph,
    MGGraph,
)
from pylibcugraph.utils cimport (
    assert_success,
    copy_to_cupy_array,
    assert_CAI_type,
    assert_AI_type,
    get_c_type_from_numpy_type,
)


def uniform_random_walks(ResourceHandle resource_handle,
                         _GPUGraph input_graph,
                         start_vertices,
                         size_t max_length):
    """
    Compute uniform random walks for each nodes in 'start_vertices'

    Parameters
    ----------
    resource_handle: ResourceHandle
        Handle to the underlying device and host resources needed for
        referencing data and running algorithms.

    input_graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    start_vertices: device array type
        Device array containing the list of starting vertices from which
        to run the uniform random walk

    max_length: size_t
        The maximum depth of the uniform random walks


    Returns
    -------
    A tuple containing two device arrays and an size_t which are respectively
    the vertices path, the edge path weights and the maximum path length

    """
    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = input_graph.c_graph_ptr

    assert_CAI_type(start_vertices, "start_vertices")

    cdef cugraph_random_walk_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef uintptr_t cai_start_ptr = \
        start_vertices.__cuda_array_interface__["data"][0]

    cdef cugraph_type_erased_device_array_view_t* start_ptr = \
        cugraph_type_erased_device_array_view_create(
            <void*>cai_start_ptr,
            len(start_vertices),
            get_c_type_from_numpy_type(start_vertices.dtype))

    error_code = cugraph_uniform_random_walks(
        c_resource_handle_ptr,
        c_graph_ptr,
        start_ptr,
        max_length,
        &result_ptr,
        &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_uniform_random_walks")

    cdef cugraph_type_erased_device_array_view_t* path_ptr = \
        cugraph_random_walk_result_get_paths(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* weights_ptr = \
        cugraph_random_walk_result_get_weights(result_ptr)

    max_path_length = \
        cugraph_random_walk_result_get_max_path_length(result_ptr)

    cupy_paths = copy_to_cupy_array(c_resource_handle_ptr, path_ptr)
    cupy_weights = copy_to_cupy_array(c_resource_handle_ptr, weights_ptr)

    cugraph_random_walk_result_free(result_ptr)
    cugraph_type_erased_device_array_view_free(start_ptr)

    return (cupy_paths, cupy_weights, max_path_length)
