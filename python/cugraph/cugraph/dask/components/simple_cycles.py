# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf
import cupy as cp
import dask_cudf
from dask.distributed import default_client, wait
from pylibcugraph import ResourceHandle
from pylibcugraph import simple_cycles as pylibcugraph_simple_cycles

import cugraph.dask.comms.comms as Comms
from cugraph.components.simple_cycles import (
    _cycles_to_frame,
    _empty_cycles_frame,
    _normalize_seeds,
    _validate_graph_and_bound,
)


def _call_simple_cycles(session_id, graph, seed_parts, dtype, bound, rank, n_workers):
    seeds = None
    if seed_parts is not None:
        # All ranks must pass a non-NULL array when filtering, even ranks with
        # no seed partitions. The C API shuffles seeds to their owning ranks.
        seeds = (
            cudf.concat(seed_parts).astype(dtype).to_cupy()
            if seed_parts
            else cp.empty(0, dtype=dtype)
        )
    return _cycles_to_frame(
        pylibcugraph_simple_cycles(
            resource_handle=ResourceHandle(Comms.get_handle(session_id).getHandle()),
            graph=graph,
            seed_vertices=seeds,
            length_bound=bound,
            do_expensive_check=True,
        ),
        rank,
        n_workers,
    )


def simple_cycles(input_graph, length_bound, seed_vertices=None):
    """Enumerate bounded simple cycles using all GPUs in a Dask cluster.

    Parameters
    ----------
    input_graph : cugraph.Graph
        Directed graph built with from_dask_cudf_edgelist. Edge weights are
        ignored. A Dask client and cuGraph communications must be initialized.
    length_bound : int
        Positive maximum cycle length. Small bounds (for example, at most 10)
        are recommended because cycle enumeration can produce large outputs.
    seed_vertices : list, cudf.Series, cudf.DataFrame, dask_cudf.Series or dask_cudf.DataFrame, optional
        External vertex IDs; return cycles containing at least one seed.
        None selects all cycles, while an empty collection selects none.
        For multi-column IDs, supply columns in source-vertex order.
        Seeds must belong to the graph.

    Returns
    -------
    dask_cudf.DataFrame
        One row per vertex occurrence with globally unique ``cycle_id``,
        zero-based ``position`` within the cycle, and external ``vertex`` IDs.
        Multi-column IDs become ``0_vertex``, ``1_vertex``, etc. Group by
        cycle_id and sort by position to recover cycles in directed cyclic
        order; the closing vertex is not repeated. Cycle IDs need not be
        contiguous and are unspecified across calls. Unrenumbering may shuffle
        rows or split a cycle across partitions.
    """
    _validate_graph_and_bound(input_graph, length_bound)
    if not isinstance(input_graph._plc_graph, dict):
        raise TypeError("input_graph must be built with from_dask_cudf_edgelist")
    client = default_client()
    workers = list(input_graph._plc_graph)
    dtype = input_graph.edgelist.edgelist_df.dtypes.iloc[0]
    seed_parts = None
    if seed_vertices is not None:
        if not isinstance(seed_vertices, (dask_cudf.Series, dask_cudf.DataFrame)):
            seed_vertices = _normalize_seeds(seed_vertices)
            seed_vertices = dask_cudf.from_cudf(seed_vertices, npartitions=len(workers))
        nulls = seed_vertices.isnull().any()
        if isinstance(seed_vertices, dask_cudf.DataFrame):
            nulls = nulls.any()
        if nulls.compute():
            raise ValueError("seed_vertices must not contain nulls")
        if len(seed_vertices) == 0:
            df = dask_cudf.from_cudf(
                _empty_cycles_frame(input_graph), npartitions=len(workers)
            )
            if input_graph.renumbered:
                df = input_graph.unrenumber(df, "vertex")
            return df
        if input_graph.renumbered:
            columns = (
                list(seed_vertices.columns)
                if isinstance(seed_vertices, dask_cudf.DataFrame)
                else None
            )
            seed_vertices = input_graph.lookup_internal_vertex_id(
                seed_vertices, columns
            )
        elif isinstance(seed_vertices, dask_cudf.DataFrame):
            if len(seed_vertices.columns) != 1:
                raise ValueError("unrenumbered graphs require single-column seeds")
            seed_vertices = seed_vertices[seed_vertices.columns[0]]
        if seed_vertices.isnull().any().compute():
            raise ValueError("seed_vertices must belong to the graph")
        seed_parts = client.compute(seed_vertices.astype(dtype).to_delayed())
        # Resolve preprocessing errors before starting collective GPU work.
        wait(seed_parts)
        for part in seed_parts:
            if part.status == "error":
                part.result()
    results = []
    try:
        for rank, worker in enumerate(workers):
            results.append(
                client.submit(
                    _call_simple_cycles,
                    Comms.get_session_id(),
                    input_graph._plc_graph[worker],
                    seed_parts[rank :: len(workers)]
                    if seed_parts is not None
                    else None,
                    dtype,
                    int(length_bound),
                    rank,
                    len(workers),
                    workers=[worker],
                    allow_other_workers=False,
                    pure=False,
                )
            )
        wait(results)
        for result in results:
            if result.status == "error":
                result.result()
        df = dask_cudf.from_delayed(results).persist()
        wait(df)
        if input_graph.renumbered:
            df = input_graph.unrenumber(df, "vertex").persist()
            wait(df)
        return df
    finally:
        for future in results + (seed_parts if seed_parts is not None else []):
            future.release()
