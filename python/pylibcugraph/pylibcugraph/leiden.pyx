# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

import math
import numbers
import warnings

from libc.stdint cimport int32_t

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
    cugraph_leiden,
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


def leiden(ResourceHandle resource_handle,
           random_state,
           _GPUGraph graph,
           max_level=None,
           double resolution=1.0,
           theta=None,
           do_expensive_check=False,
           *,
           n_iterations=2,
           beta=0.0):
    """
    Compute the modularity optimizing partition of the input graph using the
    Leiden method.

    It uses the Leiden algorithm described in:

    Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to
    Leiden: guaranteeing well-connected communities. Scientific reports,
    9(1), 5233. doi: 10.1038/s41598-019-41695-z

    Every Leiden iteration runs, at each level of the hierarchy, a local
    moving phase, a refinement phase and an aggregation of the graph by the
    refined partition, and ends with a split of every community into its
    connected components. The result has these guarantees:

    * every returned community is connected;
    * the returned modularity is the exact modularity Q(resolution) of the
      returned clustering (self-loops excluded);
    * the result is deterministic: the same graph, parameters and
      random_state give the bitwise identical clustering and modularity on
      every call, for 32- and 64-bit vertex ids, for float and double weights
      (with equal values) and, for an MGGraph, for any number of GPUs. The
      arithmetic is designed to give the same result on other GPU
      architectures as well.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    random_state : int, optional
        Seed of the hashes that randomize the local moves. Only its low 32
        bits are used. If None, defaults to a hash of process id, time, and
        hostname (see pylibcugraph.random.CuGraphRandomState). For an
        MGGraph, every rank must pass a different value; the value of rank 0
        is used on every rank.

    graph : SGGraph or MGGraph
        The input graph. It must be symmetric: every undirected edge stored
        in both directions with the same weight (the data is checked). The
        weights must be finite and non-negative; an unweighted graph uses
        weight 1 for every edge. Self-loops and edges of weight 0 are
        ignored, and parallel edges are summed. The number of vertices must
        be less than 2**30. An MGGraph is currently gathered onto every GPU,
        so it must fit on one GPU.

    max_level : int, optional (default=None)
        Deprecated and ignored: Leiden runs n_iterations iterations, each
        with as many levels as it needs. Passing a value emits a
        FutureWarning. It will be removed in a future release.

    resolution : float, optional (default=1.0)
        Called gamma in the modularity formula, this changes the size
        of the communities.  Higher resolutions lead to more smaller
        communities, lower resolutions lead to fewer larger communities.
        Must be finite and in [0, 2**20].

    theta : float, optional (default=None)
        Deprecated and ignored: it was never used by the implementation (see
        beta). Passing a value emits a FutureWarning. It will be removed in
        a future release.

    do_expensive_check : bool, optional (default=False)
        If True, performs more extensive tests on the inputs to ensure
        validity, at the expense of increased run time. The checks of the
        graph data (CSR structure, weights, symmetry) always run.

    n_iterations : int, optional (default=2)
        Keyword-only. Number of Leiden iterations (every iteration restarts
        from the partition of the previous one; Traag et al. 2019). Must be
        >= 1, or -1 to iterate until the partition is stable: until an
        iteration moves no vertex or improves the modularity by less than
        1e-6, at most 20 iterations.

    beta : float, optional (default=0.0)
        Keyword-only. Randomness of the refinement phase (theta of Traag et
        al. 2019). Reserved: only 0 is implemented, a value > 0 raises
        NotImplementedError.

    Returns
    -------
    A tuple containing the vertices, their clusters and the modularity score
    of the clustering. Clusters are numbered 0, 1, ... in order of
    decreasing size (ties broken by the smallest vertex id). For an MGGraph,
    every rank returns its local vertices and the global modularity.

    Examples
    --------
    >>> import pylibcugraph, cupy, numpy
    >>> # two triangles (0, 1, 2) and (3, 4, 5) joined by the edge (2, 3),
    >>> # every edge stored in both directions
    >>> srcs = cupy.asarray([0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3, 2, 3],
    ...                     dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 0, 2, 1, 0, 2, 4, 3, 5, 4, 3, 5, 3, 2],
    ...                     dtype=numpy.int32)
    >>> weights = cupy.ones(14, dtype=numpy.float32)
    >>> resource_handle = pylibcugraph.ResourceHandle()
    >>> graph_props = pylibcugraph.GraphProperties(
    ...     is_symmetric=True, is_multigraph=False)
    >>> G = pylibcugraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=False, renumber=False, do_expensive_check=False)
    >>> (vertices, clusters, modularity) = pylibcugraph.leiden(
    ...     resource_handle, 42, G, resolution=1.0, n_iterations=2)
    >>> vertices
    array([0, 1, 2, 3, 4, 5], dtype=int32)
    >>> clusters
    array([0, 0, 0, 1, 1, 1], dtype=int32)
    >>> round(modularity, 6)
    0.357143

    """
    # Parameters of the former implementation: still accepted, so that
    # existing (positional) calls keep working, but ignored.
    if max_level is not None:
        warnings.warn(
            "max_level is deprecated and has no effect: Leiden runs "
            "'n_iterations' iterations (default 2), each with as many levels "
            "as it needs. It will be removed in a future release.",
            FutureWarning,
        )
    if theta is not None:
        warnings.warn(
            "theta is deprecated and has no effect (it was never used by the "
            "implementation; see 'beta'). It will be removed in a future "
            "release.",
            FutureWarning,
        )

    if isinstance(n_iterations, bool) or \
            not isinstance(n_iterations, numbers.Integral):
        raise TypeError(
            "n_iterations must be an int, got "
            f"{type(n_iterations).__name__}")
    if n_iterations != -1 and not (1 <= n_iterations < 2**16):
        raise ValueError(
            "n_iterations must be >= 1 (and < 65536), or -1 to iterate until "
            f"the partition is stable, got {n_iterations}")
    if not (0.0 <= resolution <= 2.0**20):
        raise ValueError(
            f"resolution must be finite and in [0, 2**20], got {resolution}")
    beta = float(beta)
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError(f"beta must be finite and >= 0, got {beta}")
    if beta > 0.0:
        raise NotImplementedError(
            "beta > 0 (randomized refinement) is not implemented yet; use "
            "beta=0.")

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr
    cdef cugraph_hierarchical_clustering_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr
    cdef int32_t c_n_iterations = n_iterations
    cdef double c_beta = beta
    cdef bool_t c_do_expensive_check = bool(do_expensive_check)

    cg_rng_state = CuGraphRandomState(resource_handle, random_state)

    cdef cugraph_rng_state_t* rng_state_ptr = cg_rng_state.rng_state_ptr

    error_code = cugraph_leiden(c_resource_handle_ptr,
                                rng_state_ptr,
                                c_graph_ptr,
                                c_n_iterations,
                                resolution,
                                c_beta,
                                c_do_expensive_check,
                                &result_ptr,
                                &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_leiden")

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
