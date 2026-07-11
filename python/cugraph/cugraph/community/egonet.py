# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import cudf
from cugraph.utilities import ensure_cugraph_obj
from pylibcugraph import ego_graph as pylibcugraph_ego_graph
from pylibcugraph import ResourceHandle


def _convert_graph_to_output_type(G, input_type):
    """
    Given a cugraph.Graph, convert it to a new type appropriate for the
    graph algos in this module, based on input_type.
    """
    return G


def _convert_df_series_to_output_type(df, offsets, input_type):
    """
    Given a cudf.DataFrame df, convert it to a new type appropriate for the
    graph algos in this module, based on input_type.
    """
    return df, offsets


def ego_graph(
    G,
    n,
    radius=1,
    center=True,
    undirected=None,
    distance=None,
    *,
    return_offsets=False,
):
    """Compute ego graph(s) centered at one or more seed vertices.

    Parameters
    ----------
    G : cugraph.Graph, CuPy or SciPy sparse matrix
        Input graph.
    n : integer, list, cudf.Series, or cudf.DataFrame
        One seed vertex, or multiple seed vertices when ``return_offsets=True``.
    radius : integer, optional
        Include neighbors at distance less than or equal to ``radius``.
    center : bool, optional
        Defaults to True. False is not supported.
    undirected : optional
        Present for NetworkX compatibility and currently ignored.
    distance : optional
        Present for NetworkX compatibility and currently ignored.
    return_offsets : bool, keyword-only, optional
        When ``False`` (default), preserve the existing single-seed behavior and
        return a ``cugraph.Graph``. Multiple seeds raise ``ValueError`` rather
        than silently returning a composite graph.

        When ``True``, return ``(edge_dataframe, offsets)``. ``offsets[i]`` and
        ``offsets[i + 1]`` delimit the rows belonging to seed ``i``. This keeps
        the ordering and boundaries returned by pylibcugraph.

    Returns
    -------
    cugraph.Graph
        The ego graph for a single seed when ``return_offsets=False``.
    tuple
        ``(edge_dataframe, offsets)`` when ``return_offsets=True``.
    """
    (G, input_type) = ensure_cugraph_obj(G)

    if isinstance(n, int):
        n = cudf.Series([n])
    elif isinstance(n, list):
        n = cudf.Series(n)

    if isinstance(n, cudf.Series):
        seed_count = len(n)
        if G.renumbered is True:
            n = G.lookup_internal_vertex_id(n)
    elif isinstance(n, cudf.DataFrame):
        seed_count = len(n)
        if G.renumbered is True:
            n = G.lookup_internal_vertex_id(n, n.columns)
    else:
        raise TypeError(
            f"'n' must be either an integer or a list or a cudf.Series"
            f" or a cudf.DataFrame, got: {type(n)}"
        )

    if seed_count == 0:
        raise ValueError("'n' must contain at least one seed vertex")

    if seed_count > 1 and not return_offsets:
        raise ValueError(
            "Multiple seed vertices require return_offsets=True so that "
            "individual ego-graph boundaries are preserved."
        )

    n_type = G.edgelist.edgelist_df["src"].dtype
    n = n.astype(n_type)

    source, destination, weight, offsets = pylibcugraph_ego_graph(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        source_vertices=n,
        radius=radius,
        do_expensive_check=False,
    )

    df = cudf.DataFrame()
    df["src"] = source
    df["dst"] = destination
    if weight is not None:
        df["weight"] = weight

    if G.renumbered:
        df, src_names = G.unrenumber(df, "src", get_column_names=True)
        df, dst_names = G.unrenumber(df, "dst", get_column_names=True)
    else:
        src_names = "src"
        dst_names = "dst"

    if return_offsets:
        offsets = cudf.Series(offsets)
        return _convert_df_series_to_output_type(df, offsets, input_type)

    result_graph = type(G)(directed=G.is_directed())
    if G.edgelist.weights:
        result_graph.from_cudf_edgelist(
            df, source=src_names, destination=dst_names, edge_attr="weight"
        )
    else:
        result_graph.from_cudf_edgelist(df, source=src_names, destination=dst_names)

    return _convert_graph_to_output_type(result_graph, input_type)
