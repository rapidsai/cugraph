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
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.community_algorithms cimport (
    cugraph_hierarchical_clustering_result_t,
    cugraph_ecg,
    cugraph_hierarchical_clustering_result_get_vertices,
    cugraph_hierarchical_clustering_result_get_clusters,
    cugraph_hierarchical_clustering_result_get_modularity,
    cugraph_hierarchical_clustering_result_free,
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

def ecg(ResourceHandle resource_handle,
        random_state,
        _GPUGraph graph,
        double min_weight,
        size_t ensemble_size,
        size_t max_level,
        double threshold,
        double resolution,
        bool_t do_expensive_check
        ):
    """
    Compute the Ensemble Clustering for Graphs (ECG) partition of the input
    graph. ECG runs truncated Louvain on an ensemble of permutations of the
    input graph, then uses the ensemble partitions to determine weights for
    the input graph. The final result is found by running full Louvain on
    the input graph using the determined weights.

    See https://arxiv.org/abs/1809.05578 for further information.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    random_state : int , optional
        Random state to use when generating samples. Optional argument,
        defaults to a hash of process id, time, and hostname.
        (See pylibcugraph.random.CuGraphRandomState)

    graph : SGGraph
        The input graph.

    min_weight : double, optional (default=0.5)
        The minimum value to assign as an edgeweight in the ECG algorithm.
        It should be a value in the range [0,1] usually left as the default
        value of .05

    ensemble_size : size_t, optional (default=16)
        The number of graph permutations to use for the ensemble.
        The default value is 16, larger values may produce higher quality
        partitions for some graphs.

    max_level: size_t
        This controls the maximum number of levels/iterations of the leiden
        algorithm. When specified the algorithm will terminate after no more
        than the specified number of iterations. No error occurs when the
        algorithm terminates early in this manner.

    threshold: float
        Modularity gain threshold for each level. If the gain of
        modularity between 2 levels of the algorithm is less than the
        given threshold then the algorithm stops and returns the
        resulting communities.

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
    A tuple containing the hierarchical clustering vertices, clusters

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
    >>> (vertices, clusters) = pylibcugraph.ecg(resource_handle, G)
    # FIXME: Check this docstring example
    >>> vertices
    [0, 1, 2]
    >>> clusters
    [0, 0, 0]

    """

    if min_weight is None:
        min_weight = 0.5
    if ensemble_size is None:
        ensemble_size = 16

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr
    cdef cugraph_hierarchical_clustering_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cg_rng_state = CuGraphRandomState(resource_handle, random_state)

    cdef cugraph_rng_state_t* rng_state_ptr = cg_rng_state.rng_state_ptr

    error_code = cugraph_ecg(c_resource_handle_ptr,
                             rng_state_ptr,
                             c_graph_ptr,
                             min_weight,
                             ensemble_size,
                             max_level,
                             threshold,
                             resolution,
                             do_expensive_check,
                             &result_ptr,
                             &error_ptr)

    assert_success(error_code, error_ptr, "cugraph_ecg")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef cugraph_type_erased_device_array_view_t* vertices_ptr = \
        cugraph_hierarchical_clustering_result_get_vertices(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* clusters_ptr = \
        cugraph_hierarchical_clustering_result_get_clusters(result_ptr)
    cdef double modularity = \
        cugraph_hierarchical_clustering_result_get_modularity(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_clusters = copy_to_cupy_array(c_resource_handle_ptr, clusters_ptr)

    cugraph_hierarchical_clustering_result_free(result_ptr)

    return (cupy_vertices, cupy_clusters, modularity)
