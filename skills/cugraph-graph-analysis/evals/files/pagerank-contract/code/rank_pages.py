# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Intentionally flawed PageRank implementation for the evaluation task."""


def rank_pages(edges, all_pages, *, alpha=0.85, max_iter=200, tol=1.0e-6):
    import cugraph

    graph = cugraph.Graph(directed=True)
    graph.from_cudf_edgelist(
        edges,
        source="src_page",
        destination="dst_page",
        edge_attr="latency_ms",
        renumber=False,
    )

    ranks = cugraph.pagerank(
        graph,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        fail_on_nonconvergence=False,
    )
    return ranks.sort_values("pagerank", ascending=False)
