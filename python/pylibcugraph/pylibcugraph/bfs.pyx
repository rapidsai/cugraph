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
from libc.stdint cimport int32_t
from libc.limits cimport INT_MAX

from pylibcugraph.resource_handle cimport ResourceHandle
from pylibcugraph._cugraph_c.algorithms cimport (
    cugraph_bfs,
    cugraph_paths_result_t,
    cugraph_paths_result_get_vertices,
    cugraph_paths_result_get_predecessors,
    cugraph_paths_result_get_distances,
    cugraph_paths_result_free,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view,
    cugraph_type_erased_device_array_t,
    cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_device_array_view_create,
    cugraph_type_erased_device_array_view_free,
)
from pylibcugraph._cugraph_c.resource_handle cimport (
    bool_t,
    data_type_id_t,
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph.utils cimport (
    assert_success,
    copy_to_cupy_array,
    assert_CAI_type,
    get_c_type_from_numpy_type,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph.graphs cimport (
    _GPUGraph,
)

def bfs(ResourceHandle handle, _GPUGraph graph, 
        sources, bool_t direction_optimizing, int32_t depth_limit, 
        bool_t compute_predecessors, bool_t do_expensive_check):
    """
    Performs a Breadth-first search starting from the provided sources.
    Returns the distances, and predecessors if requested.

    Parameters
    ----------
    handle: ResourceHandle
        The resource handle responsible for managing device resources
        that this algorithm will use
    graph: SGGraph or MGGraph
        The graph to operate upon
    sources: cudf.Series
        The vertices to start the breadth-first search from.  Should
        match the numbering of the provided graph.  All workers must
        have a unique set of sources. Empty sets are allowed as long
        as at least one worker has a source.
    direction_optimizing: bool_t
        Whether to treat the graph as undirected (should only be called
        on a symmetric graph)
    depth_limit: int32_t
        The depth limit at which the traversal will be stopped.  If this
        is a negative number, the traversal will run without a depth limit.
    compute_predecessors: bool_t
        Whether to compute the predecessors.  If left blank, -1 will be
        returned instead of the correct predecessor of each vertex.
    do_expensive_check : bool_t
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    A tuple of device arrays (cupy arrays) of the form
    (distances, predecessors, vertices)

    Examples
    --------

    M = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
                      dtype=['int32', 'int32', 'float32'], header=None)
    G = cugraph.Graph()
    G.from_cudf_edgelist(M, source='0', destination='1', edge_attr='2')

    handle = ResourceHandle()

    srcs = G.edgelist.edgelist_df['src']
    dsts = G.edgelist.edgelist_df['dst']
    weights = G.edgelist.edgelist_df['weights']

    sg = SGGraph(
        resource_handle = handle, 
        graph_properties = GraphProperties(is_multigraph=G.is_multigraph()), 
        src_array = srcs, 
        dst_array = dsts, 
        weight_array = weights,
        store_transposed=False,
        renumber=False,
        do_expensive_check=do_expensive_check
    )

    res = pylibcugraph_bfs(
            handle,    
            sg,
            cudf.Series([0], dtype='int32'),
            False,
            10,
            True,
            False
    )

    distances, predecessors, vertices = res
    
    final_results = cudf.DataFrame({
        'distance': cudf.Series(distances),
        'vertex': cudf.Series(vertices),
        'predecessor': cudf.Series(predecessors),
    })

    """

    try:
        import cupy
    except ModuleNotFoundError:
        raise RuntimeError("bfs requires the cupy package, which could not "
                           "be imported")
    assert_CAI_type(sources, "sources")

    if depth_limit <= 0:
        depth_limit = INT_MAX - 1

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef uintptr_t cai_sources_ptr = \
        sources.__cuda_array_interface__["data"][0]

    cdef cugraph_type_erased_device_array_view_t* sources_view_ptr = \
        cugraph_type_erased_device_array_view_create(
            <void*>cai_sources_ptr,
            len(sources),
            get_c_type_from_numpy_type(sources.dtype))
    
    cdef cugraph_paths_result_t* result_ptr

    error_code = cugraph_bfs(
        c_resource_handle_ptr,
        c_graph_ptr,
        sources_view_ptr,
        direction_optimizing,
        depth_limit,
        compute_predecessors,
        do_expensive_check,
        &result_ptr,
        &error_ptr
    )
    assert_success(error_code, error_ptr, "cugraph_bfs")

    # Extract individual device array pointers from result
    cdef cugraph_type_erased_device_array_view_t* distances_ptr = \
        cugraph_paths_result_get_distances(result_ptr)

    cdef cugraph_type_erased_device_array_view_t* predecessors_ptr = \
        cugraph_paths_result_get_predecessors(result_ptr)
    
    cdef cugraph_type_erased_device_array_view_t* vertices_ptr = \
        cugraph_paths_result_get_vertices(result_ptr)

    # copy to cupy arrays
    cupy_distances = copy_to_cupy_array(c_resource_handle_ptr, distances_ptr)
    cupy_predecessors = copy_to_cupy_array(c_resource_handle_ptr, predecessors_ptr)
    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    
    # deallocate the no-longer needed result struct
    cugraph_paths_result_free(result_ptr)

    return (cupy_distances, cupy_predecessors, cupy_vertices)
