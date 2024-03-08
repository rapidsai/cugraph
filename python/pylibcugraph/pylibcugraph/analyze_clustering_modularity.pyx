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
    cugraph_type_erased_device_array_view_free,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.community_algorithms cimport (
    cugraph_clustering_result_t,
    cugraph_analyze_clustering_modularity,
)

from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)
from pylibcugraph.graphs cimport (
    _GPUGraph,
)
from pylibcugraph.utils cimport (
    assert_success,
    create_cugraph_type_erased_device_array_view_from_py_obj
)


def analyze_clustering_modularity(ResourceHandle resource_handle,
                                  _GPUGraph graph,
                                  size_t num_clusters,
                                  vertex,
                                  cluster,
                                  ):
    """
    Compute modularity score of the specified clustering.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph
        The input graph.

    num_clusters : size_t
        Specifies the number of clusters to find, must be greater than 1.

    vertex : device array type
        Vertex ids from the clustering to analyze.

    cluster : device array type
        Cluster ids from the clustering to analyze.

    Returns
    -------
    The modularity score of the specified clustering.

    Examples
    --------
    >>> import pylibcugraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 0], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibcugraph.ResourceHandle()
    >>> graph_props = pylibcugraph.GraphProperties(
    ...     is_symmetric=True, is_multigraph=False)
    >>> G = pylibcugraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> (vertex, cluster) = pylibcugraph.spectral_modularity_maximization(
    ...     resource_handle, G, num_clusters=5, num_eigen_vects=2, evs_tolerance=0.00001
    ...     evs_max_iter=100, kmean_tolerance=0.00001, kmean_max_iter=100)
        # FIXME: Fix docstring result.
    >>> vertices
    ############
    >>> clusters
    ############
    >>> score = pylibcugraph.analyze_clustering_modularity(
    ...     resource_handle, G, num_clusters=5, vertex=vertex, cluster=cluster)
    >>> score
    ############


    """

    cdef double score = 0

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr
    cdef cugraph_clustering_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef cugraph_type_erased_device_array_view_t* \
        vertex_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                vertex)

    cdef cugraph_type_erased_device_array_view_t* \
        cluster_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                cluster)


    error_code = cugraph_analyze_clustering_modularity(c_resource_handle_ptr,
                                                       c_graph_ptr,
                                                       num_clusters,
                                                       vertex_view_ptr,
                                                       cluster_view_ptr,
                                                       &score,
                                                       &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_analyze_clustering_modularity")

    if vertex is not None:
        cugraph_type_erased_device_array_view_free(vertex_view_ptr)
    if cluster is not None:
        cugraph_type_erased_device_array_view_free(cluster_view_ptr)

    return score
