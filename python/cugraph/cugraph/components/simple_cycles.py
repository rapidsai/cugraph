# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numbers import Integral

import cudf
import cupy as cp
from pylibcugraph import ResourceHandle
from pylibcugraph import simple_cycles as pylibcugraph_simple_cycles

from cugraph.structure import Graph


def _validate_graph_and_bound(G, length_bound):
    if not isinstance(G, Graph):
        raise TypeError("G must be a cugraph.Graph")
    if not G.is_directed():
        raise ValueError("input graph must be directed")
    if isinstance(length_bound, bool) or not isinstance(length_bound, Integral):
        raise TypeError("length_bound must be an integer")
    if length_bound <= 0:
        raise ValueError("length_bound must be greater than 0")
    # The C API converts the bound to the graph's vertex type.
    if length_bound > cp.iinfo(G.edgelist.edgelist_df.dtypes.iloc[0]).max:
        raise ValueError("length_bound exceeds the maximum value for the vertex type")


def _normalize_seeds(seed_vertices):
    if isinstance(seed_vertices, list):
        seed_vertices = cudf.Series(seed_vertices)
    if not isinstance(seed_vertices, (cudf.Series, cudf.DataFrame)):
        raise TypeError("seed_vertices must be a list, cudf.Series or cudf.DataFrame")
    if isinstance(seed_vertices, cudf.DataFrame):
        if seed_vertices.isnull().any().any():
            raise ValueError("seed_vertices must not contain nulls")
    elif seed_vertices.isnull().any():
        raise ValueError("seed_vertices must not contain nulls")
    return seed_vertices


def _cycles_to_frame(result, rank=0, n_workers=1):
    vertices, offsets, num_cycles = result
    lengths = cp.diff(offsets).astype(cp.int64)
    df = cudf.DataFrame()
    # Interleaved IDs distinguish cycles from different ranks without gathering
    # cycle counts on the client. IDs need not be contiguous.
    df["cycle_id"] = cp.repeat(
        cp.arange(num_cycles, dtype=cp.int64) * n_workers + rank, lengths
    )
    df["position"] = cp.arange(len(vertices), dtype=cp.int64) - cp.repeat(
        offsets[:-1].astype(cp.int64), lengths
    )
    df["vertex"] = vertices
    return df


def _empty_cycles_frame(G):
    return _cycles_to_frame(
        (
            cp.empty(0, dtype=G.edgelist.edgelist_df.dtypes.iloc[0]),
            cp.zeros(1, dtype=cp.uint64),
            0,
        )
    )


def simple_cycles(G, length_bound, seed_vertices=None):
    """Enumerate simple cycles of bounded length in a directed graph.

    Parameters
    ----------
    G : cugraph.Graph
        Single-GPU directed graph. Edge weights are ignored. Undirected graphs
        are unsupported.
    length_bound : int
        Maximum cycle length, greater than zero. Small bounds (for example,
        at most 10) are recommended: the number of cycles can grow rapidly.
    seed_vertices : list, cudf.Series or cudf.DataFrame, optional
        External vertex IDs. Return only cycles containing at least one seed.
        None selects all cycles; an empty collection selects no cycles.
        Use a DataFrame with columns in source-vertex order for multi-column
        vertex IDs. Seeds must belong to the graph.

    Returns
    -------
    cudf.DataFrame
        One row per vertex occurrence, with ``cycle_id``, ``position`` and
        ``vertex`` columns. Group by cycle_id and sort by position to recover
        each cycle in directed cyclic order. The closing vertex is not repeated.
        Cycle IDs and enumeration order are unspecified. Renumbered vertices
        are returned in external ID space; multi-column IDs use the usual
        ``0_vertex``, ``1_vertex``, etc. columns.

    Examples
    --------
    >>> import cudf
    >>> import cugraph
    >>> G = cugraph.Graph(directed=True)
    >>> G.from_cudf_edgelist(
    ...     cudf.DataFrame({"src": [0, 1, 2], "dst": [1, 2, 0]}),
    ...     source="src", destination="dst")
    >>> cycles = cugraph.simple_cycles(G, length_bound=3)
    >>> cycles.sort_values(["cycle_id", "position"])
    """
    _validate_graph_and_bound(G, length_bound)
    if isinstance(G._plc_graph, dict):
        raise TypeError("use cugraph.dask.simple_cycles for a distributed graph")
    if seed_vertices is not None:
        seed_vertices = _normalize_seeds(seed_vertices)
        # libcugraph's expensive checks require a nonempty global seed set.
        if len(seed_vertices) == 0:
            df = _empty_cycles_frame(G)
            return G.unrenumber(df, "vertex") if G.renumbered else df
        if G.renumbered:
            columns = (
                list(seed_vertices.columns)
                if isinstance(seed_vertices, cudf.DataFrame)
                else None
            )
            seed_vertices = G.lookup_internal_vertex_id(seed_vertices, columns)
        elif isinstance(seed_vertices, cudf.DataFrame):
            if len(seed_vertices.columns) != 1:
                raise ValueError("unrenumbered graphs require single-column seeds")
            seed_vertices = seed_vertices.iloc[:, 0]
        if seed_vertices.isnull().any():
            raise ValueError("seed_vertices must belong to the graph")
        seed_vertices = seed_vertices.astype(
            G.edgelist.edgelist_df.dtypes.iloc[0]
        ).to_cupy()
    df = _cycles_to_frame(
        pylibcugraph_simple_cycles(
            resource_handle=ResourceHandle(),
            graph=G._plc_graph,
            seed_vertices=seed_vertices,
            length_bound=int(length_bound),
            do_expensive_check=True,
        )
    )
    if G.renumbered:
        df = G.unrenumber(df, "vertex")
    return df
