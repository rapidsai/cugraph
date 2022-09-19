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
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.community_algorithms cimport (
    cugraph_heirarchical_clustering_result_t,
    cugraph_louvain,
    cugraph_heirarchical_clustering_result_get_vertices,
    cugraph_heirarchical_clustering_result_get_clusters,
    cugraph_heirarchical_clustering_result_free,
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


def EXPERIMENTAL__louvain(ResourceHandle resource_handle,
                          _GPUGraph graph,
                          size_t max_level,
                          double resolution,
                          bool_t do_expensive_check):
    """
    Compute the modularity optimizing partition of the input graph using the
    Louvain method.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph.

    max_level: size_t
        This controls the maximum number of levels/iterations of the Louvain
        algorithm. When specified the algorithm will terminate after no more
        than the specified number of iterations. No error occurs when the
        algorithm terminates early in this manner.

    resolution: double
        Called gamma in the modularity formula, this changes the size
        of the communities.  Higher resolutions lead to more smaller
        communities, lower resolutions lead to fewer larger communities.
        Defaults to 1.

    do_expensive_check : bool_t
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    # FIXME: no modularity_score is returned liek in the legacy implementation
    A tuple of device arrays, conataining the heirarchical clustering vertices
    and clusters.

    Examples
    --------
    # FIXME: No example yet

    """
    # FIXME: import these modules here for now until a better pattern can be
    # used for optional imports (perhaps 'import_optional()' from cugraph), or
    # these are made hard dependencies.
    try:
        import cupy
    except ModuleNotFoundError:
        raise RuntimeError("louvain requires the cupy package, which could "
                           "not be imported")
    try:
        import numpy
    except ModuleNotFoundError:
        raise RuntimeError("louvain requires the numpy package, which could "
                           "not be imported")

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr
    cdef cugraph_heirarchical_clustering_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    error_code = cugraph_louvain(c_resource_handle_ptr,
                                 c_graph_ptr,
                                 max_level,
                                 resolution,
                                 do_expensive_check,
                                 &result_ptr,
                                 &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_louvain")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef cugraph_type_erased_device_array_view_t* vertices_ptr = \
        cugraph_heirarchical_clustering_result_get_vertices(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* clusters_ptr = \
        cugraph_heirarchical_clustering_result_get_clusters(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_clusters = copy_to_cupy_array(c_resource_handle_ptr, clusters_ptr)

    cugraph_heirarchical_clustering_result_free(result_ptr)

    return (cupy_vertices, cupy_clusters)
