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
    subgraph_monomorphism,
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

    result_df = subgraph_monomorphism(G, pattern_G)

    expected = nx_monomorphisms_set(cugraph_to_nx(G), cugraph_to_nx(pattern_G))
    assert result_to_set(result_df) == expected


@pytest.mark.sg
def test_matches_networkx_on_small_handbuilt_graph():
    # bowtie: two triangles sharing vertex 2, plus a pendant vertex
    target_edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2), (4, 5)]
    G = build_cugraph_from_edges(target_edges)

    for pattern_name, pattern_edges in PATTERNS.items():
        pattern_G = build_cugraph_from_edges(pattern_edges)
        result_df = subgraph_monomorphism(G, pattern_G)
        expected = nx_monomorphisms_set(cugraph_to_nx(G), cugraph_to_nx(pattern_G))
        assert result_to_set(result_df) == expected, pattern_name


@pytest.mark.sg
def test_triangle_count_on_karate():
    G = karate.get_graph(download=True)
    pattern_G = build_cugraph_from_edges(PATTERNS["triangle"])

    result_df = subgraph_monomorphism(G, pattern_G)

    n_triangles = sum(nx.triangles(cugraph_to_nx(G)).values()) // 3
    # each undirected triangle appears as 3! = 6 ordered embeddings
    assert len(result_df) == 6 * n_triangles


@pytest.mark.sg
def test_embeddings_are_valid():
    G = karate.get_graph(download=True)
    pattern_edges = PATTERNS["4-cycle"]
    pattern_G = build_cugraph_from_edges(pattern_edges)

    result_df = subgraph_monomorphism(G, pattern_G)
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

    result_df = subgraph_monomorphism(G, pattern_G)

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
        subgraph_monomorphism(dG, pattern_G)
    with pytest.raises(ValueError):
        subgraph_monomorphism(G, dG)


@pytest.mark.sg
def test_disconnected_pattern_raises():
    G = karate.get_graph(download=True)
    pattern_G = build_cugraph_from_edges([(0, 1), (2, 3)])

    with pytest.raises(ValueError):
        subgraph_monomorphism(G, pattern_G)


@pytest.mark.sg
def test_pattern_larger_than_target_raises():
    G = build_cugraph_from_edges([(0, 1), (1, 2)])
    pattern_G = build_cugraph_from_edges(PATTERNS["K4"])

    with pytest.raises(ValueError):
        subgraph_monomorphism(G, pattern_G)


@pytest.mark.sg
def test_pattern_self_loop_raises():
    G = karate.get_graph(download=True)
    pattern_G = build_cugraph_from_edges([(0, 0), (0, 1)])

    with pytest.raises(ValueError, match="self-loop"):
        subgraph_monomorphism(G, pattern_G)


@pytest.mark.sg
def test_target_self_loops_are_ignored():
    # Identical results with and without self-loops on the target.
    base_edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
    G_clean = build_cugraph_from_edges(base_edges)
    G_loops = build_cugraph_from_edges(base_edges + [(0, 0), (3, 3)])
    pattern_G = build_cugraph_from_edges(PATTERNS["triangle"])

    assert result_to_set(subgraph_monomorphism(G_loops, pattern_G)) == (
        result_to_set(subgraph_monomorphism(G_clean, pattern_G))
    )


@pytest.mark.sg
def test_pattern_with_noncontiguous_ids():
    # Pattern vertex ids are arbitrary labels; result columns are named by
    # the original ids, and matches equal those of the relabeled pattern.
    G = karate.get_graph(download=True)
    pattern_G = build_cugraph_from_edges([(5, 17), (17, 42)])

    result_df = subgraph_monomorphism(G, pattern_G)
    assert list(result_df.columns) == ["5", "17", "42"]
    expected = nx_monomorphisms_set(cugraph_to_nx(G), cugraph_to_nx(pattern_G))
    assert result_to_set(result_df) == expected


@pytest.mark.sg
def test_no_match_returns_empty_dataframe():
    # tree target has no triangles
    G = build_cugraph_from_edges([(0, 1), (1, 2), (1, 3), (3, 4)])
    pattern_G = build_cugraph_from_edges(PATTERNS["triangle"])

    result_df = subgraph_monomorphism(G, pattern_G)
    assert len(result_df) == 0
    assert list(result_df.columns) == ["0", "1", "2"]


@pytest.mark.sg
def test_empty_result_keeps_unmatched_new_vertex_columns():
    # Star target: partial path embeddings (leaf-center-leaf) exist, but no
    # injective embedding of a longer path does, so a mid-assembly merge
    # filters down to zero rows. The empty result must still carry one
    # column per pattern vertex, including vertices introduced by motifs
    # merged at or after the point the intermediate became empty (P4), and
    # merges after that point must still find their join columns (P5).
    star_edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    G = build_cugraph_from_edges(star_edges)

    for path_edges, n_vertices in (
        ([(0, 1), (1, 2), (2, 3)], 4),
        ([(0, 1), (1, 2), (2, 3), (3, 4)], 5),
    ):
        pattern_G = build_cugraph_from_edges(path_edges)
        result_df = subgraph_monomorphism(G, pattern_G)
        assert len(result_df) == 0
        assert list(result_df.columns) == [str(v) for v in range(n_vertices)]


@pytest.mark.sg
def test_partition_writer_splits_oversized_chunks():
    # A single filtered chunk larger than the row limit must be split so
    # that every emitted partition honors the limit.
    from cugraph.experimental.isomorphism.solver import _PartitionWriter

    writer = _PartitionWriter(row_limit=10)
    writer.add(cudf.DataFrame({"a": range(35)}))
    writer.add(cudf.DataFrame({"a": range(35, 42)}))
    parts = writer.finish()
    assert all(len(part) <= 10 for part in parts)
    combined = cudf.concat(parts)
    assert sorted(combined["a"].to_pandas()) == list(range(42))

    # An oversized chunk that is an exact multiple of the limit, arriving
    # while the buffer is non-empty.
    writer = _PartitionWriter(row_limit=10)
    writer.add(cudf.DataFrame({"a": range(3)}))
    writer.add(cudf.DataFrame({"a": range(3, 33)}))
    parts = writer.finish()
    assert all(len(part) <= 10 for part in parts)
    combined = cudf.concat(parts)
    assert sorted(combined["a"].to_pandas()) == list(range(33))


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
    baseline_df = subgraph_monomorphism(G, pattern_G)

    monkeypatch.setattr(
        _MotifSubgraphIsomorphismSolver, "_JOIN_MEM_BUDGET_FRACTION", 0.0
    )
    monkeypatch.setattr(
        _MotifSubgraphIsomorphismSolver, "_JOIN_MEM_BUDGET_MIN_GIB", 1e-6
    )
    monkeypatch.setattr(_MotifSubgraphIsomorphismSolver, "_FANOUT_SAMPLE_ROWS", 8)
    result_df = subgraph_monomorphism(G, pattern_G)
    assert result_to_set(result_df) == result_to_set(baseline_df)


@pytest.mark.sg
def test_multi_partition_results_give_identical_results(monkeypatch):
    # Force intermediate and final results to span many partitions so the
    # multi-partition (host NumPy) assembly path is exercised; in normal
    # operation it only triggers for results beyond ~1.8B rows. Every
    # partition emitted anywhere in the solve must honor the row limit.
    from cugraph.experimental.isomorphism import solver as solver_mod

    G = karate.get_graph(download=True)
    pattern_G = build_cugraph_from_edges(PATTERNS["4-cycle"])
    baseline_df = subgraph_monomorphism(G, pattern_G)

    emitted_partition_sizes = []

    class RecordingWriter(solver_mod._PartitionWriter):
        def finish(self):
            parts = super().finish()
            emitted_partition_sizes.extend(len(part) for part in parts)
            return parts

    monkeypatch.setattr(solver_mod._MotifSubgraphIsomorphismSolver, "_ROW_LIMIT", 100)
    monkeypatch.setattr(solver_mod, "_PartitionWriter", RecordingWriter)
    result_df = subgraph_monomorphism(G, pattern_G)
    assert result_to_set(result_df) == result_to_set(baseline_df)
    # End-to-end: all partition output flows through the writer, and every
    # emitted partition satisfies the configured limit.
    assert len(emitted_partition_sizes) > 1
    assert all(size <= 100 for size in emitted_partition_sizes)


@pytest.mark.sg
def test_choose_batch_rows_bounds_skewed_partitions(monkeypatch):
    # A partition whose leading rows match nothing must not be merged in a
    # single whole-partition batch: the unsampled tail may join a hub key,
    # and an unbounded batch could overflow cuDF's row limit at scale. The
    # strided sample (or, failing that, the byte-budget fallback) must
    # return a batch smaller than the partition under a tiny budget.
    from cugraph.experimental.isomorphism.solver import (
        _MotifSubgraphIsomorphismSolver,
    )

    monkeypatch.setattr(_MotifSubgraphIsomorphismSolver, "_FANOUT_SAMPLE_ROWS", 8)
    solver = _MotifSubgraphIsomorphismSolver(
        cudf.DataFrame({"src": [0], "dst": [1]}), 2
    )
    # 64 non-matching rows followed by 64 rows all joining key 0, which has
    # fan-out 8 in next_df.
    left_df = cudf.DataFrame({"m0_v0": [1] * 128, "m0_v1": [99] * 64 + [0] * 64})
    next_df = cudf.DataFrame({"m1_v0": [0] * 8, "m1_v1": range(8)})
    batch_rows = solver._choose_batch_rows(
        left_df, next_df, ["m0_v1"], ["m1_v0"], budget_bytes=64
    )
    assert 1 <= batch_rows < len(left_df)


@pytest.mark.sg
def test_malformed_motifs_raise():
    with pytest.raises(ValueError):
        MotifData(name="empty", motif=[])
    with pytest.raises(ValueError):
        # vertices must be contiguous 0..k-1
        MotifData(name="shifted", motif=[(1, 2)])


@pytest.mark.sg
def test_user_motifs_with_multi_partition_results(monkeypatch):
    # Precomputed motif embeddings pass through the solver's output path;
    # with a small _ROW_LIMIT that output is a multi-partition (host NumPy)
    # array, exercising the NumPy branch of the motif-table construction.
    from cugraph.experimental.isomorphism.solver import (
        _MotifSubgraphIsomorphismSolver,
    )

    G = karate.get_graph(download=True)
    pattern_G = build_cugraph_from_edges(PATTERNS["K4"])
    baseline_df = subgraph_monomorphism(G, pattern_G)

    monkeypatch.setattr(_MotifSubgraphIsomorphismSolver, "_ROW_LIMIT", 100)
    result_df = subgraph_monomorphism(
        G, pattern_G, motifs=[MotifData(name="M3-path", motif=[(0, 1), (1, 2)])]
    )
    assert result_to_set(result_df) == result_to_set(baseline_df)


@pytest.mark.sg
def test_motif_library_gives_identical_results():
    G = karate.get_graph(download=True)
    pattern_G = build_cugraph_from_edges(PATTERNS["K4"])

    baseline_df = subgraph_monomorphism(G, pattern_G)
    result_df = subgraph_monomorphism(G, pattern_G, motifs=default_motif_library())
    assert result_to_set(result_df) == result_to_set(baseline_df)


@pytest.mark.sg
def test_user_supplied_motifs_give_identical_results():
    # Motifs passed directly as MotifData objects, without going through
    # default_motif_library().
    G = karate.get_graph(download=True)
    pattern_G = build_cugraph_from_edges(PATTERNS["K4"])
    baseline_df = subgraph_monomorphism(G, pattern_G)

    motifs = [
        MotifData(name="M3-path", motif=[(0, 1), (1, 2)]),
        MotifData(name="M3-triangle", motif=[(0, 1), (1, 2), (0, 2)]),
        MotifData(name="M4-star", motif=[(0, 1), (0, 2), (0, 3)]),
    ]
    result_df = subgraph_monomorphism(G, pattern_G, motifs=motifs)
    assert result_to_set(result_df) == result_to_set(baseline_df)

    # Supplying the pattern itself as a motif also works: the decomposition
    # then covers the pattern with a single slice.
    result_df = subgraph_monomorphism(
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

    result_df = subgraph_monomorphism(G, pattern_G)

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
