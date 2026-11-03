# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Tuple, Union

from pylibcugraph import leiden as pylibcugraph_leiden
from pylibcugraph import ResourceHandle
from cugraph.structure import Graph
import cudf


def leiden(
    G: Graph,
    max_iter: Union[int, None] = None,
    resolution: float = 1.0,
    random_state: Union[int, None] = None,
    theta: Union[float, None] = None,
    *,
    n_iterations: int = 2,
    beta: float = 0.0,
) -> Tuple[cudf.DataFrame, float]:
    """
    Compute the modularity optimizing partition of the input graph using the
    Leiden algorithm

    It uses the Leiden method described in:

    Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden:
    guaranteeing well-connected communities. Scientific reports, 9(1), 5233.
    doi: 10.1038/s41598-019-41695-z

    Every Leiden iteration runs, at each level of the hierarchy, a local
    moving phase, a refinement phase and an aggregation of the graph by the
    refined partition, and ends with a split of every community into its
    connected components. The result has these guarantees:

    * every returned community is connected;
    * the returned modularity is the exact modularity of the returned
      partition (see Returns);
    * the result is deterministic: with a fixed `random_state`, the same
      graph gives the bitwise identical partition and modularity on every
      call, for 32- and 64-bit vertex ids and for float and double weights
      (with equal values). The arithmetic is designed to give the same
      result on other GPU architectures as well.

    Parameters
    ----------
    G : cugraph.Graph
        cuGraph graph descriptor of type Graph

        The current implementation only supports undirected graphs. The
        weights must be finite and non-negative; an unweighted graph uses
        weight 1 for every edge. Self-loops and edges of weight 0 are
        ignored, and the parallel edges of a MultiGraph are summed. The
        number of vertices must be less than 2**30.

        The adjacency list will be computed if not already present.

    max_iter : integer, optional (default=None)
        Deprecated and ignored: Leiden runs `n_iterations` iterations, each
        with as many levels as it needs, so this cap on the number of levels
        is no longer needed. Passing a value emits a FutureWarning. It will
        be removed in a future release.

    resolution: float, optional (default=1.0)
        Called gamma in the modularity formula, this changes the size
        of the communities.  Higher resolutions lead to more smaller
        communities, lower resolutions lead to fewer larger communities.
        Must be finite and in [0, 2**20]. Defaults to 1.

    random_state: int, optional(default=None)
        Seed of the hashes that randomize the order of the local moves. Only
        its low 32 bits are used. The partition depends on the seed and on
        the internal vertex numbering of `G` (with
        ``G.from_cudf_edgelist(..., renumber=False)`` the input vertex ids
        are used). Defaults to a hash of process id, time, and hostname. The
        same graph, parameters and seed give the bitwise identical result.

    theta: float, optional (default=None)
        Deprecated and ignored: it was never used by the implementation (see
        `beta`). Passing a value emits a FutureWarning. It will be removed
        in a future release.

    n_iterations: int, optional (default=2)
        Keyword-only. Number of Leiden iterations (every iteration restarts
        from the partition of the previous one; Traag et al. 2019). Must be
        >= 1, or -1 to iterate until the partition is stable: until an
        iteration moves no vertex or improves the modularity by less than
        1e-6, at most 20 iterations. Two iterations are recommended.

    beta: float, optional (default=0.0)
        Keyword-only. Randomness of the refinement phase (theta of Traag et
        al. 2019). Reserved: only 0 (merge along the best edge) is implemented;
        a value > 0 raises NotImplementedError.

    Returns
    -------
    parts : cudf.DataFrame
        GPU data frame of size V containing two columns the vertex id and the
        partition id it is assigned to.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['partition'] : cudf.Series
            Contains the partition assigned to the vertices. Partitions are
            numbered 0, 1, ... in order of decreasing size.

    modularity_score : float
        The modularity of the partition, computed exactly (in double
        precision) for the given resolution:
        Q = sum_c [ L_c / (2m) - resolution * (K_c / (2m))**2 ], where L_c
        is twice the weight of the edges inside partition c, K_c the total
        degree of c and 2m the total degree, self-loops excluded.

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> parts, modularity_score = cugraph.leiden(G, random_state=42)

    """

    if G.is_directed():
        raise ValueError("input graph must be undirected")

    # Parameters of the former implementation: still accepted, so that
    # existing (positional) calls keep working, but ignored.
    if max_iter is not None:
        warnings.warn(
            "max_iter is deprecated and has no effect: Leiden runs "
            "'n_iterations' iterations (default 2), each with as many levels "
            "as it needs. It will be removed in a future release.",
            FutureWarning,
            stacklevel=2,
        )
    if theta is not None:
        warnings.warn(
            "theta is deprecated and has no effect (it was never used by the "
            "implementation; see 'beta'). It will be removed in a future "
            "release.",
            FutureWarning,
            stacklevel=2,
        )

    vertex, partition, modularity_score = pylibcugraph_leiden(
        resource_handle=ResourceHandle(),
        random_state=random_state,
        graph=G._plc_graph,
        resolution=resolution,
        do_expensive_check=False,
        n_iterations=n_iterations,
        beta=beta,
    )

    df = cudf.DataFrame()
    df["vertex"] = vertex
    df["partition"] = partition

    if G.renumbered:
        parts = G.unrenumber(df, "vertex")
    else:
        parts = df

    return parts, modularity_score
