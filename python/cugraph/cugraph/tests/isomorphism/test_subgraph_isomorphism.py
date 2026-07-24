# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc

import cudf
import networkx as nx
import pytest
from networkx.algorithms.isomorphism import GraphMatcher

import cugraph
from cugraph.datasets import email_Eu_core, karate
from cugraph.experimental import (
    MotifData,
    default_motif_library,
    subgraph_isomorphism,
)

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# =============================================================================
# Helpers
# =============================================================================

PATTERNS = {
    "triangle": [(0, 1), (1, 2), (2, 0)],
    "P3-path": [(0, 1), (1, 2)],
    "4-cycle": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "K4": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
}


def build_cugraph_from_edges(edges):
    df = cudf.DataFrame({"src": [u for u, v in edges], "dst": [v for u, v in edges]})
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(df, source="src", destination="dst")
    return G


def cugraph_to_nx(G):
    edge_df = G.view_edge_list().to_pandas()
    src_col, dst_col = edge_df.columns[0], edge_df.columns[1]
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(zip(edge_df[src_col], edge_df[dst_col]))
    return nx_graph


def result_to_set(result_df):
    """Rows of the result as a set of (pattern_vertex, target_vertex) tuple
    tuples, independent of row order."""
    pattern_vertices = [int(c) for c in result_df.columns]
    rows = result_df.to_pandas().itertuples(index=False)
    return {tuple(sorted(zip(pattern_vertices, (int(v) for v in row)))) for row in rows}


def nx_monomorphisms_set(target_nx, pattern_nx):
    matcher = GraphMatcher(target_nx, pattern_nx)
    return {
        tuple(sorted((int(p), int(t)) for t, p in mapping.items()))
        for mapping in matcher.subgraph_monomorphisms_iter()
    }


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.sg
@pytest.mark.parametrize("pattern_name", list(PATTERNS.keys()))
def test_matches_networkx_monomorphisms_on_karate(pattern_name):
    G = karate.get_graph(download=True)
    pattern_G = build_cugraph_from_edges(PATTERNS[pattern_name])

    result_df = subgraph_isomorphism(G, pattern_G)

    expected = nx_monomorphisms_set(cugraph_to_nx(G), cugraph_to_nx(pattern_G))
    assert result_to_set(result_df) == expected


@pytest.mark.sg
def test_matches_networkx_on_small_handbuilt_graph():
    # bowtie: two triangles sharing vertex 2, plus a pendant vertex
    target_edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2), (4, 5)]
    G = build_cugraph_from_edges(target_edges)

    for pattern_name, pattern_edges in PATTERNS.items():
        pattern_G = build_cugraph_from_edges(pattern_edges)
        result_df = subgraph_isomorphism(G, pattern_G)
        expected = nx_monomorphisms_set(cugraph_to_nx(G), cugraph_to_nx(pattern_G))
        assert result_to_set(result_df) == expected, pattern_name


@pytest.mark.sg
def test_triangle_count_on_karate():
    G = karate.get_graph(download=True)
    pattern_G = build_cugraph_from_edges(PATTERNS["triangle"])

    result_df = subgraph_isomorphism(G, pattern_G)

    n_triangles = sum(nx.triangles(cugraph_to_nx(G)).values()) // 3
    # each undirected triangle appears as 3! = 6 ordered embeddings
    assert len(result_df) == 6 * n_triangles


@pytest.mark.sg
def test_embeddings_are_valid():
    G = karate.get_graph(download=True)
    pattern_edges = PATTERNS["4-cycle"]
    pattern_G = build_cugraph_from_edges(pattern_edges)

    result_df = subgraph_isomorphism(G, pattern_G)
    assert len(result_df) > 0

    target_nx = cugraph_to_nx(G)
    pattern_vertices = [int(c) for c in result_df.columns]
    for row in result_df.to_pandas().itertuples(index=False):
        mapping = dict(zip(pattern_vertices, (int(v) for v in row)))
        # injective
        assert len(set(mapping.values())) == len(mapping)
        # every pattern edge maps to a target edge
        for u, v in pattern_edges:
            assert target_nx.has_edge(mapping[u], mapping[v])


@pytest.mark.sg
def test_renumbering_with_noncontiguous_ids():
    # bowtie graph with non-contiguous, shifted vertex ids
    target_edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)]
    shifted_edges = [(u * 10 + 5, v * 10 + 5) for u, v in target_edges]
    G = build_cugraph_from_edges(shifted_edges)
    pattern_G = build_cugraph_from_edges(PATTERNS["triangle"])

    result_df = subgraph_isomorphism(G, pattern_G)

    expected = nx_monomorphisms_set(cugraph_to_nx(G), cugraph_to_nx(pattern_G))
    assert result_to_set(result_df) == expected
    # output must contain the original (shifted) ids
    returned_ids = set()
    for col in result_df.columns:
        returned_ids.update(int(v) for v in result_df[col].to_pandas())
    assert returned_ids.issubset({u * 10 + 5 for u in range(5)})


@pytest.mark.sg
def test_directed_graph_raises():
    df = cudf.DataFrame({"src": [0, 1], "dst": [1, 2]})
    dG = cugraph.Graph(directed=True)
    dG.from_cudf_edgelist(df, source="src", destination="dst")
    pattern_G = build_cugraph_from_edges([(0, 1)])
    G = build_cugraph_from_edges([(0, 1), (1, 2)])

    with pytest.raises(ValueError):
        subgraph_isomorphism(dG, pattern_G)
    with pytest.raises(ValueError):
        subgraph_isomorphism(G, dG)


@pytest.mark.sg
def test_disconnected_pattern_raises():
    G = karate.get_graph(download=True)
    pattern_G = build_cugraph_from_edges([(0, 1), (2, 3)])

    with pytest.raises(ValueError):
        subgraph_isomorphism(G, pattern_G)


@pytest.mark.sg
def test_pattern_larger_than_target_raises():
    G = build_cugraph_from_edges([(0, 1), (1, 2)])
    pattern_G = build_cugraph_from_edges(PATTERNS["K4"])

    with pytest.raises(ValueError):
        subgraph_isomorphism(G, pattern_G)


@pytest.mark.sg
def test_no_match_returns_empty_dataframe():
    # tree target has no triangles
    G = build_cugraph_from_edges([(0, 1), (1, 2), (1, 3), (3, 4)])
    pattern_G = build_cugraph_from_edges(PATTERNS["triangle"])

    result_df = subgraph_isomorphism(G, pattern_G)
    assert len(result_df) == 0
    assert list(result_df.columns) == ["0", "1", "2"]


@pytest.mark.sg
def test_tiny_join_budget_gives_identical_results(monkeypatch):
    # Force many small streamed-join batches (and multiple partition-writer
    # flushes) through the public API by shrinking the adaptive memory
    # budget; results must be identical to the default single-batch solve.
    from cugraph.experimental.isomorphism.solver import (
        _MotifSubgraphIsomorphismSolver,
    )

    G = karate.get_graph(download=True)
    pattern_G = build_cugraph_from_edges(PATTERNS["4-cycle"])
    baseline_df = subgraph_isomorphism(G, pattern_G)

    monkeypatch.setattr(
        _MotifSubgraphIsomorphismSolver, "_JOIN_MEM_BUDGET_FRACTION", 0.0
    )
    monkeypatch.setattr(
        _MotifSubgraphIsomorphismSolver, "_JOIN_MEM_BUDGET_MIN_GIB", 1e-6
    )
    monkeypatch.setattr(_MotifSubgraphIsomorphismSolver, "_FANOUT_SAMPLE_ROWS", 8)
    result_df = subgraph_isomorphism(G, pattern_G)
    assert result_to_set(result_df) == result_to_set(baseline_df)


@pytest.mark.sg
def test_multi_partition_results_give_identical_results(monkeypatch):
    # Force intermediate and final results to span many partitions so the
    # multi-partition (host NumPy) assembly path is exercised; in normal
    # operation it only triggers for results beyond ~1.8B rows.
    from cugraph.experimental.isomorphism.solver import (
        _MotifSubgraphIsomorphismSolver,
    )

    G = karate.get_graph(download=True)
    pattern_G = build_cugraph_from_edges(PATTERNS["4-cycle"])
    baseline_df = subgraph_isomorphism(G, pattern_G)

    monkeypatch.setattr(_MotifSubgraphIsomorphismSolver, "_ROW_LIMIT", 100)
    result_df = subgraph_isomorphism(G, pattern_G)
    assert result_to_set(result_df) == result_to_set(baseline_df)


@pytest.mark.sg
def test_motif_library_gives_identical_results():
    G = karate.get_graph(download=True)
    pattern_G = build_cugraph_from_edges(PATTERNS["K4"])

    baseline_df = subgraph_isomorphism(G, pattern_G)
    result_df = subgraph_isomorphism(G, pattern_G, motifs=default_motif_library())
    assert result_to_set(result_df) == result_to_set(baseline_df)


@pytest.mark.sg
def test_user_supplied_motifs_give_identical_results():
    # Motifs passed directly as MotifData objects, without going through
    # default_motif_library().
    G = karate.get_graph(download=True)
    pattern_G = build_cugraph_from_edges(PATTERNS["K4"])
    baseline_df = subgraph_isomorphism(G, pattern_G)

    motifs = [
        MotifData(name="M3-path", motif=[(0, 1), (1, 2)]),
        MotifData(name="M3-triangle", motif=[(0, 1), (1, 2), (0, 2)]),
        MotifData(name="M4-star", motif=[(0, 1), (0, 2), (0, 3)]),
    ]
    result_df = subgraph_isomorphism(G, pattern_G, motifs=motifs)
    assert result_to_set(result_df) == result_to_set(baseline_df)

    # Supplying the pattern itself as a motif also works: the decomposition
    # then covers the pattern with a single slice.
    result_df = subgraph_isomorphism(
        G, pattern_G, motifs=[MotifData(name="K4", motif=PATTERNS["K4"])]
    )
    assert result_to_set(result_df) == result_to_set(baseline_df)


@pytest.mark.sg
def test_large_graph_triangle_count():
    # A denser, larger target than karate (~1k vertices, ~25k edges,
    # ~100k triangles); validated by count against nx.triangles plus
    # validity spot-checks, since full NetworkX enumeration would be slow.
    G = email_Eu_core.get_graph(download=True, ignore_weights=True)
    pattern_G = build_cugraph_from_edges(PATTERNS["triangle"])

    result_df = subgraph_isomorphism(G, pattern_G)

    target_nx = cugraph_to_nx(G)
    target_nx.remove_edges_from(nx.selfloop_edges(target_nx))
    n_triangles = sum(nx.triangles(target_nx).values()) // 3
    # each undirected triangle appears as 3! = 6 ordered embeddings
    assert len(result_df) == 6 * n_triangles

    # validity spot-check on a sample of embeddings
    pattern_vertices = [int(c) for c in result_df.columns]
    sample = result_df.head(500).to_pandas()
    for row in sample.itertuples(index=False):
        mapping = dict(zip(pattern_vertices, (int(v) for v in row)))
        assert len(set(mapping.values())) == len(mapping)
        for u, v in PATTERNS["triangle"]:
            assert target_nx.has_edge(mapping[u], mapping[v])
