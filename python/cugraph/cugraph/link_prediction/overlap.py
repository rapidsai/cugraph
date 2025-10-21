# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cugraph.utilities import (
    ensure_valid_dtype,
    renumber_vertex_pair,
)
import cudf
import warnings

from pylibcugraph import (
    overlap_coefficients as pylibcugraph_overlap_coefficients,
    all_pairs_overlap_coefficients as pylibcugraph_all_pairs_overlap_coefficients,
)
from pylibcugraph import ResourceHandle

from cugraph.structure import Graph


def overlap_coefficient(
    G: Graph,
    ebunch: cudf.DataFrame = None,
    do_expensive_check: bool = False,  # deprecated
) -> cudf.DataFrame:
    """
    Compute overlap coefficient.

    Parameters
    ----------
    G : cugraph.Graph
        cuGraph Graph instance, should contain the connectivity
        information as an edge list. The graph should be undirected where an
        undirected edge is represented by a directed edge in both direction.
        The adjacency list will be computed if not already present.

        This implementation only supports undirected, non-multi edge Graph.

    ebunch : cudf.DataFrame of node pairs, optional (default=None)
        A GPU dataframe consisting of two columns representing pairs of
        vertices (u, v) where u and v are nodes in the graph.

        If provided, the Overlap coefficient is computed for the given vertex
        pairs. Otherwise, the current implementation computes the overlap
        coefficient for all vertices that are two hops apart in the graph.

    do_expensive_check : bool, optional (default=False)
        Deprecated.
        This option added a check to ensure integer vertex IDs are sequential
        values from 0 to V-1. That check is now redundant because cugraph
        unconditionally renumbers and un-renumbers integer vertex IDs for
        optimal performance, therefore this option is deprecated and will be
        removed in a future version.

    Returns
    -------
    df  : cudf.DataFrame
        GPU data frame of size E (the default) or the size of the given pairs
        (first, second) containing the overlap weights. The ordering is
        relative to the adjacency list, or that given by the specified vertex
        pairs.

        ddf['first']: dask_cudf.Series
            The first vertex ID of each pair (will be identical to first if specified).
        ddf['second']: dask_cudf.Series
            The second vertex ID of each pair (will be identical to second if
            specified).
        ddf['overlap_coeff']: dask_cudf.Series
            The computed overlap coefficient between the first and the second
            vertex ID.

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> from cugraph import overlap_coefficient
    >>> G = karate.get_graph(download=True, ignore_weights=True)
    >>> df = overlap_coefficient(G)
    """
    warnings.warn(
        "deprecated as of 25.10. Use `overlap()` instead. "
        "If calling with a NetworkX Graph object, use networkx with the "
        "nx-cugraph backend. See: https://rapids.ai/nx-cugraph",
        DeprecationWarning,
    )

    if do_expensive_check:
        warnings.warn(
            "do_expensive_check is deprecated since vertex IDs are no longer "
            "required to be consecutively numbered",
            FutureWarning,
        )

    vertex_pair = ebunch

    df = overlap(G, vertex_pair)

    return df


def overlap(
    input_graph: Graph,
    vertex_pair: cudf.DataFrame = None,
    use_weight: bool = False,
) -> cudf.DataFrame:
    """
    Compute the Overlap Coefficient between each pair of vertices connected by
    an edge, or between arbitrary pairs of vertices specified by the user.
    Overlap Coefficient is defined between two sets as the ratio of the volume
    of their intersection over the smaller of their two volumes. In the
    context of graphs, the neighborhood of a vertex is seen as a set. The
    Overlap Coefficient weight of each edge represents the strength of
    connection between vertices based on the relative similarity of their
    neighbors. If first is specified but second is not, or vice versa, an
    exception will be thrown.

    cugraph.overlap, in the absence of a specified vertex pair list, will
    compute the two_hop_neighbors of the entire graph to construct a vertex pair
    list and will return the overlap coefficient for those vertex pairs. This is
    not advisable as the vertex_pairs can grow exponentially with respect to the
    size of the datasets

    Parameters
    ----------
    input_graph : cugraph.Graph
        cuGraph Graph instance, should contain the connectivity information
        as an edge list. The adjacency list will be computed if not already
        present.

        This implementation only supports undirected, non-multi edge Graph.

    vertex_pair : cudf.DataFrame, optional (default=None)
        A GPU dataframe consisting of two columns representing pairs of
        vertices. If provided, the overlap coefficient is computed for the
        given vertex pairs, else, it is computed for all vertex pairs.

    use_weight : bool, optional (default=False)
        Flag to indicate whether to compute weighted overlap (if use_weight==True)
        or un-weighted overlap (if use_weight==False).
        'input_graph' must be weighted if 'use_weight=True'.


    Returns
    -------
    df : cudf.DataFrame
        GPU data frame of size E (the default) or the size of the given pairs
        (first, second) containing the Overlap coefficients. The ordering is
        relative to the adjacency list, or that given by the specified vertex
        pairs.

        df['first'] : cudf.Series
            The first vertex ID of each pair (will be identical to first if specified).
        df['second'] : cudf.Series
            The second vertex ID of each pair (will be identical to second if
            specified).
        df['overlap_coeff'] : cudf.Series
            The computed overlap coefficient between the first and the second
            vertex ID.

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> from cugraph import overlap
    >>> input_graph = karate.get_graph(download=True, ignore_weights=True)
    >>> df = overlap(input_graph)

    """

    if input_graph.is_directed():
        raise ValueError("Input must be an undirected Graph.")

    if vertex_pair is None:
        # Call two_hop neighbor of the entire graph
        vertex_pair = input_graph.get_two_hop_neighbors()

    v_p_num_col = len(vertex_pair.columns)

    if isinstance(vertex_pair, cudf.DataFrame):
        vertex_pair = renumber_vertex_pair(input_graph, vertex_pair)
        vertex_pair = ensure_valid_dtype(input_graph, vertex_pair)
        src_col_name = vertex_pair.columns[0]
        dst_col_name = vertex_pair.columns[1]
        first = vertex_pair[src_col_name]
        second = vertex_pair[dst_col_name]

    elif vertex_pair is not None:
        raise ValueError("vertex_pair must be a cudf dataframe")

    first, second, overlap_coeff = pylibcugraph_overlap_coefficients(
        resource_handle=ResourceHandle(),
        graph=input_graph._plc_graph,
        first=first,
        second=second,
        use_weight=use_weight,
        do_expensive_check=False,
    )

    if input_graph.renumbered:
        vertex_pair = input_graph.unrenumber(
            vertex_pair, src_col_name, preserve_order=True
        )
        vertex_pair = input_graph.unrenumber(
            vertex_pair, dst_col_name, preserve_order=True
        )

    if v_p_num_col == 2:
        # single column vertex
        vertex_pair = vertex_pair.rename(
            columns={src_col_name: "first", dst_col_name: "second"}
        )

    df = vertex_pair
    df["overlap_coeff"] = cudf.Series(overlap_coeff)

    return df


def all_pairs_overlap(
    input_graph: Graph,
    vertices: cudf.Series = None,
    use_weight: bool = False,
    topk: int = None,
) -> cudf.DataFrame:
    """
    Compute the All Pairs Overlap Coefficient between each pair of vertices connected
    by an edge, or between arbitrary pairs of vertices specified by the user.
    Overlap Coefficient is defined between two sets as the ratio of the volume
    of their intersection over the smaller of their two volumes. In the
    context of graphs, the neighborhood of a vertex is seen as a set. The
    Overlap Coefficient weight of each edge represents the strength of
    connection between vertices based on the relative similarity of their
    neighbors.

    cugraph.all_pairs_overlap, in the absence of specified vertices, will
    compute the two_hop_neighbors of the entire graph to construct a vertex pair
    list and will return the overlap coefficient for all the vertex pairs in the graph.
    This is not advisable as the vertex_pairs can grow exponentially with respect to
    the size of the datasets.

    If the topk parameter is specified then the result will only contain the top k
    highest scoring results.

    Parameters
    ----------
    input_graph : cugraph.Graph
        cuGraph Graph instance, should contain the connectivity information
        as an edge list. The graph should be undirected where an undirected
        edge is represented by a directed edge in both direction.The adjacency
        list will be computed if not already present.

        This implementation only supports undirected, non-multi Graphs.

    vertices : int or list or cudf.Series or cudf.DataFrame, optional (default=None)
        A GPU Series containing the input vertex list.  If the vertex list is not
        provided then the current implementation computes the overlap coefficient for
        all vertices that are two hops apart in the graph.

    use_weight : bool, optional (default=False)
        Flag to indicate whether to compute weighted overlap (if use_weight==True)
        or un-weighted overlap (if use_weight==False).
        'input_graph' must be weighted if 'use_weight=True'.

    topk : int, optional (default=None)
        Specify the number of answers to return otherwise returns the entire
        solution

    Returns
    -------
    df  : cudf.DataFrame
        GPU data frame of size E (the default) or the size of the given pairs
        (first, second) containing the Overlap weights. The ordering is
        relative to the adjacency list, or that given by the specified vertex
        pairs.

        df['first'] : cudf.Series
            The first vertex ID of each pair (will be identical to first if specified).
        df['second'] : cudf.Series
            The second vertex ID of each pair (will be identical to second if
            specified).
        df['overlap_coeff'] : cudf.Series
            The computed Overlap coefficient between the first and the second
            vertex ID.

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> from cugraph import all_pairs_overlap
    >>> input_graph = karate.get_graph(download=True, ignore_weights=True)
    >>> df = all_pairs_overlap(input_graph)

    """
    if input_graph.is_directed():
        raise ValueError("Input must be an undirected Graph.")

    if vertices is not None:

        if isinstance(vertices, int):
            vertices = [vertices]

        if isinstance(vertices, list):
            vertices = cudf.Series(
                vertices,
                dtype=input_graph.edgelist.edgelist_df[input_graph.srcCol].dtype,
            )

        if input_graph.renumbered is True:
            if isinstance(vertices, cudf.DataFrame):
                vertices = input_graph.lookup_internal_vertex_id(
                    vertices, vertices.columns
                )
            else:
                vertices = input_graph.lookup_internal_vertex_id(vertices)

    first, second, overlap_coeff = pylibcugraph_all_pairs_overlap_coefficients(
        resource_handle=ResourceHandle(),
        graph=input_graph._plc_graph,
        vertices=vertices,
        use_weight=use_weight,
        topk=topk,
        do_expensive_check=False,
    )
    vertex_pair = cudf.DataFrame()
    vertex_pair["first"] = first
    vertex_pair["second"] = second

    if input_graph.renumbered:
        vertex_pair = input_graph.unrenumber(vertex_pair, "first", preserve_order=True)
        vertex_pair = input_graph.unrenumber(vertex_pair, "second", preserve_order=True)

    df = vertex_pair
    df["overlap_coeff"] = cudf.Series(overlap_coeff)

    return df
