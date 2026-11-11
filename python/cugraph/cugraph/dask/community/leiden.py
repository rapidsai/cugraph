# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import warnings

from dask.distributed import wait, default_client
import cugraph.dask.comms.comms as Comms
import dask_cudf
import dask
from dask import delayed
import cudf

from pylibcugraph import ResourceHandle
from pylibcugraph import leiden as pylibcugraph_leiden
import numpy
import cupy as cp
from typing import Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from cugraph import Graph


def convert_to_cudf(result: cp.ndarray) -> Tuple[cudf.DataFrame, float]:
    """
    Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
    """
    cupy_vertex, cupy_partition, modularity = result
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertex
    df["partition"] = cupy_partition

    return df, modularity


def _call_plc_leiden(
    sID: bytes,
    mg_graph_x,
    resolution: float,
    random_state: Union[int, None],
    n_iterations: int,
    beta: float,
    do_expensive_check: bool,
) -> Tuple[cp.ndarray, cp.ndarray, float]:
    # Every rank needs a different seed (cugraph_rng_state_create checks it),
    # and the algorithm uses the seed of rank 0 on every rank: rank r passes
    # random_state + r, so the effective seed is random_state.
    if random_state is not None:
        random_state = random_state + Comms.get_worker_id(sID)
    return pylibcugraph_leiden(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        random_state=random_state,
        graph=mg_graph_x,
        resolution=resolution,
        do_expensive_check=do_expensive_check,
        n_iterations=n_iterations,
        beta=beta,
    )


def leiden(
    input_graph: Graph,
    max_iter: Union[int, None] = None,
    resolution: float = 1.0,
    random_state: Union[int, None] = None,
    theta: Union[float, None] = None,
    *,
    n_iterations: int = 2,
    beta: float = 0.0,
) -> Tuple[dask_cudf.DataFrame, float]:
    """
    Compute the modularity optimizing partition of the input graph using the
    Leiden method

    Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden:
    guaranteeing well-connected communities. Scientific reports, 9(1), 5233.
    doi: 10.1038/s41598-019-41695-z

    This runs the same algorithm as `cugraph.leiden`, with the same
    guarantees: every returned community is connected, the returned
    modularity is exact, and the result is deterministic. The first level is
    split by rows over the workers; the coarser levels run on the first
    worker, which needs about the workspace of a single-GPU run. Graphs small
    enough for replicas (at most 2**18 counted edges, i.e. stored entries
    without self-loops and zero weights, and fewer than 2**29 vertices), and
    every graph on more than 32 workers, are gathered onto every worker
    instead. Given the internal vertex numbering of the graph, the result
    does not depend on the number of workers: it is bitwise identical to
    `cugraph.leiden` on a graph with the same internal numbering and the
    same `random_state`. A distributed graph is always
    renumbered internally (also with ``renumber=False``), so
    `cugraph.dask.leiden` and `cugraph.leiden` on the same input generally
    return different, equally valid partitions.

    Parameters
    ----------
    input_graph : cugraph.Graph
        The graph descriptor should contain the connectivity information
        and weights. The adjacency list will be computed if not already
        present.
        The current implementation only supports undirected graphs. The
        weights must be finite and non-negative; an unweighted graph uses
        weight 1 for every edge. Self-loops and edges of weight 0 are
        ignored, and the parallel edges of a MultiGraph are summed. The
        number of vertices must be less than 2**30 - 1.

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
        its low 32 bits are used, and it is the same for every worker. The
        partition depends on the seed and on the internal vertex numbering
        of `input_graph`; repeated calls on the same graph with the same
        seed return the same partition. Defaults to a hash of process id,
        time, and hostname.

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
    parts : dask_cudf.DataFrame
        GPU data frame of size V containing two columns the vertex id and the
        partition id it is assigned to.

        ddf['vertex'] : cudf.Series
            Contains the vertex identifiers
        ddf['partition'] : cudf.Series
            Contains the partition assigned to the vertices. Partitions are
            numbered 0, 1, ... in order of decreasing size.

    modularity_score : float
        The modularity of the partition, computed exactly (in double
        precision) for the given resolution, self-loops excluded (see
        `cugraph.leiden`).

    Examples
    --------
    >>> import cugraph.dask as dcg
    >>> import dask_cudf
    >>> # ... Init a DASK Cluster
    >>> #    see https://docs.nvidia.com/cugraph/latest/dask-cugraph.html
    >>> # Download dataset from https://github.com/rapidsai/cugraph/datasets/..
    >>> chunksize = dcg.get_chunksize(datasets_path / "karate.csv")
    >>> ddf = dask_cudf.read_csv(datasets_path / "karate.csv",
    ...                          blocksize=chunksize, delimiter=" ",
    ...                          names=["src", "dst", "value"],
    ...                          dtype=["int32", "int32", "float32"])
    >>> dg = cugraph.Graph()
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst')
    >>> parts, modularity_score = dcg.leiden(dg, random_state=42)

    """

    if input_graph.is_directed():
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

    # Return a client if one has started
    client = default_client()

    do_expensive_check = False

    # Every worker gets the same parameters and random_state (see
    # _call_plc_leiden for the per-rank seed).
    result = [
        client.submit(
            _call_plc_leiden,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            resolution,
            random_state,
            n_iterations,
            beta,
            do_expensive_check,
            workers=[w],
            allow_other_workers=False,
        )
        for w in input_graph._plc_graph
    ]

    wait(result)

    part_mod_score = [client.submit(convert_to_cudf, r) for r in result]
    wait(part_mod_score)

    vertex_dtype = input_graph.edgelist.edgelist_df.dtypes.iloc[0]
    empty_df = cudf.DataFrame(
        {
            "vertex": numpy.empty(shape=0, dtype=vertex_dtype),
            "partition": numpy.empty(shape=0, dtype="int32"),
        }
    )

    part_mod_score = [delayed(lambda x: x, nout=2)(r) for r in part_mod_score]

    ddf = dask_cudf.from_delayed(
        [r[0] for r in part_mod_score], meta=empty_df, verify_meta=False
    ).persist()

    mod_score = dask.array.from_delayed(
        part_mod_score[0][1], shape=(1,), dtype=float
    ).compute()

    wait(ddf)
    wait(mod_score)

    wait([r.release() for r in part_mod_score])

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "vertex")

    return ddf, mod_score
