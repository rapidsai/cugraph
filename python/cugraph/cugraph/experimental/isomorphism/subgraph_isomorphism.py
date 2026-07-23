# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cudf

from cugraph.experimental.isomorphism.solver import (
    _MotifSubgraphIsomorphismSolver,
)
from cugraph.utilities.utils import import_optional

nx = import_optional("networkx")


def _pattern_to_nx(pattern_G):
    """Convert a small cugraph.Graph pattern into a networkx.Graph on the
    original (unrenumbered) vertex ids."""
    edge_df = pattern_G.view_edge_list().to_pandas()
    # view_edge_list may keep the user's original column names; take the
    # source/destination columns positionally.
    src_col, dst_col = edge_df.columns[0], edge_df.columns[1]

    if (edge_df[src_col] == edge_df[dst_col]).any():
        raise ValueError(
            "Pattern graph must not contain self-loops: a self-loop can "
            "never be matched since target self-loops are ignored."
        )

    pattern_nx = nx.Graph()
    pattern_nx.add_nodes_from(pattern_G.nodes().to_pandas())
    pattern_nx.add_edges_from(zip(edge_df[src_col], edge_df[dst_col]))
    return pattern_nx


def EXPERIMENTAL__subgraph_isomorphism(G, pattern_G, motifs=None):
    """
    Find all subgraph isomorphisms (monomorphisms) of a pattern graph in a
    target graph using GPU-accelerated motif-based decomposition. Algortithm
    described in:    
        Wang, Y., Ginez, E., Friel, J., Baum, Y., Kim, J. S., Shih, A, O. Green, 
        “Δ-Motif: Parallel Subgraph Isomorphism via Tabular Operations for Scalable Layout Selection”, 
        IEEE Quantum Week (QCE), 2026

    The pattern is decomposed into small motifs (CPU VF2 on the pattern
    only), each motif's embeddings in the target are computed as cuDF
    tables, and full embeddings are assembled with cuDF joins plus an
    overlap-consistency filter. The joins are streamed in adaptively sized
    batches (scaled to free device memory) and intermediate results are
    held as partitions below cuDF's 2**31 - 1 column-size limit, so
    intermediate solution sets may exceed that limit. For problems whose
    intermediates exceed GPU memory entirely, enable cuDF spilling or RMM
    managed memory in your application before calling (e.g.
    ``cudf.set_option("spill", True)`` or
    ``rmm.reinitialize(managed_memory=True)``).

    Note on semantics: the returned embeddings are *monomorphisms*, i.e., 
    every pattern edge maps to a target edge and vertices map injectively, 
    but non-adjacent pattern vertices are allowed to map to adjacent target
    vertices. 
    This matches NetworkX ``GraphMatcher.subgraph_monomorphisms_iter``,
    not the induced ``subgraph_isomorphisms_iter``.

    Parameters
    ----------
    G : cugraph.Graph
        Undirected target graph. Self-loops are ignored.

    pattern_G : cugraph.Graph
        Undirected, connected pattern graph to search for. Must not contain
        self-loops and may not have more vertices or edges than the target.

    motifs : list of MotifData, optional (default=None)
        Optional motif building blocks used to decompose the pattern (see
        ``cugraph.experimental.MotifData`` and ``default_motif_library``).
        The single-edge M2 motif is always included automatically, so the
        default finds embeddings edge by edge. Larger motifs can speed up
        big patterns, but each one costs a full pre-solve to enumerate its
        embeddings in the target.

    Returns
    -------
    result : cudf.DataFrame
        One row per embedding found. One column per pattern vertex, named by
        the pattern vertex id (as a string, sorted order); values are the
        target vertex ids (original ids if the graph was renumbered). An
        empty DataFrame (with the same columns) means no embedding exists.

    Examples
    --------
    >>> import cudf
    >>> from cugraph import Graph
    >>> from cugraph.datasets import karate
    >>> from cugraph.experimental import subgraph_isomorphism
    >>> G = karate.get_graph(download=True)
    >>> triangle = cudf.DataFrame(
    ...     {"src": [0, 1, 2], "dst": [1, 2, 0]}
    ... )
    >>> pattern_G = Graph()
    >>> pattern_G.from_cudf_edgelist(triangle, source="src",
    ...                              destination="dst")
    >>> mappings = subgraph_isomorphism(G, pattern_G)

    """
    if G.is_directed() or pattern_G.is_directed():
        raise ValueError("input graphs must be undirected")

    if G.edgelist is None:
        # Materialize an edge list (e.g. for graphs built from an adjacency
        # list).
        G.view_edge_list()

    # Work in the renumbered internal vertex space: the solver expects
    # compact 0..n-1 ids. Do not use view_edge_list() here (it unrenumbers).
    edge_df = G.edgelist.edgelist_df[["src", "dst"]]
    edge_df = edge_df[edge_df["src"] != edge_df["dst"]]
    num_vertices = G.number_of_vertices()

    pattern_nx = _pattern_to_nx(pattern_G)

    solver = _MotifSubgraphIsomorphismSolver(edge_df, num_vertices)
    solver.generate_graph_motif_data(motifs)
    result = solver.solve(pattern_nx)

    vertex_dtype = G.edgelist.edgelist_df["src"].dtype
    if result is None:
        column_names = [str(v) for v in sorted(pattern_nx.nodes)]
        return cudf.DataFrame(
            {name: cudf.Series([], dtype=vertex_dtype) for name in column_names}
        )

    column_names = [str(v) for v in result.pattern_vertices]
    result_df = cudf.DataFrame(result.mappings, columns=column_names)
    # Cast up from the solver's compact uint dtypes so unrenumber's merge
    # against the renumber map does not hit a dtype mismatch.
    for col in column_names:
        result_df[col] = result_df[col].astype(vertex_dtype)

    if G.renumbered:
        for col in column_names:
            result_df = G.unrenumber(result_df, col)

    return result_df
