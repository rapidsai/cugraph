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
import warnings

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
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.core_algorithms cimport (   
    cugraph_core_result_t,
    cugraph_k_core_result_t,
    cugraph_core_result_create,
    cugraph_k_core,
    cugraph_k_core_degree_type_t,
    cugraph_k_core_result_get_src_vertices,
    cugraph_k_core_result_get_dst_vertices,
    cugraph_k_core_result_get_weights,
    cugraph_k_core_result_free,
    cugraph_core_result_free,
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

def k_core(ResourceHandle resource_handle,
           _GPUGraph graph,
           size_t k,
           degree_type,
           core_result,
           bool_t do_expensive_check):
    """
    Compute the k-core of the graph G
    A k-core of a graph is a maximal subgraph that
    contains nodes of degree k or more. This call does not support a graph
    with self-loops and parallel edges.

    Parameters
    ----------
    resource_handle: ResourceHandle
        Handle to the underlying device and host resource needed for
        referencing data and running algorithms.
    
    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.
    
    k : size_t (default=None)
        Order of the core. This value must not be negative. If set to None
        the main core is returned.
    
    degree_type: str
        This option determines if the core number computation should be based
        on input, output, or both directed edges, with valid values being
        "incoming", "outgoing", and "bidirectional" respectively.
        This option is currently ignored in this release, and setting it will
        result in a warning.
    
    core_result : device array type
        Precomputed core number of the nodes of the graph G
        If set to None, the core numbers of the nodes are calculated
        internally.

    do_expensive_check: bool
        If True, performs more extensive tests on the inputs to ensure
        validity, at the expense of increased run time.

    Returns
    -------
    A tuple of device arrays contaning the sources, destinations vertices
    and the weights.

    Examples
    --------
    # FIXME: No example yet

    """
    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_core_result_t* core_result_ptr
    cdef cugraph_k_core_result_t* k_core_result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr


    degree_type_map = {
        "incoming": cugraph_k_core_degree_type_t.K_CORE_DEGREE_TYPE_IN,
        "outgoing": cugraph_k_core_degree_type_t.K_CORE_DEGREE_TYPE_OUT,
        "bidirectional": cugraph_k_core_degree_type_t.K_CORE_DEGREE_TYPE_INOUT}

    cdef cugraph_type_erased_device_array_view_t* \
        vertices_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                core_result["vertex"])
    
    cdef cugraph_type_erased_device_array_view_t* \
        core_numbers_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                core_result["values"])
    
    # Create a core_number result
    error_code = cugraph_core_result_create(c_resource_handle_ptr,
                                            vertices_view_ptr,
                                            core_numbers_view_ptr,
                                            &core_result_ptr,
                                            &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_core_result_create")


    # compute k_core
    error_code = cugraph_k_core(c_resource_handle_ptr,
                                c_graph_ptr,
                                k,
                                degree_type_map[degree_type],
                                core_result_ptr,
                                do_expensive_check,
                                &k_core_result_ptr,
                                &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_k_core_number")


    cdef cugraph_type_erased_device_array_view_t* src_vertices_ptr = \
        cugraph_k_core_result_get_src_vertices(k_core_result_ptr)
    cdef cugraph_type_erased_device_array_view_t* dst_vertices_ptr = \
        cugraph_k_core_result_get_dst_vertices(k_core_result_ptr)
    cdef cugraph_type_erased_device_array_view_t* weigths_ptr = \
        cugraph_k_core_result_get_weights(k_core_result_ptr)

    cupy_src_vertices = copy_to_cupy_array(c_resource_handle_ptr, src_vertices_ptr)
    cupy_dst_vertices = copy_to_cupy_array(c_resource_handle_ptr, dst_vertices_ptr)
    cupy_weights = copy_to_cupy_array(c_resource_handle_ptr, weigths_ptr)

    cugraph_k_core_result_free(k_core_result_ptr)
    cugraph_core_result_free(core_result_ptr)

    return (cupy_src_vertices, cupy_dst_vertices, cupy_weights)
