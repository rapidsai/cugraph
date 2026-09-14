# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Intentionally flawed component analysis for the evaluation task."""


def component_views(edges, all_services):
    import cugraph

    graph = cugraph.Graph(directed=True)
    graph.from_cudf_edgelist(
        edges,
        source="caller",
        destination="callee",
        renumber=True,
    )

    weak = cugraph.weakly_connected_components(graph, directed=False)
    strong = cugraph.strongly_connected_components(graph, directed=True)
    return weak, strong


def partitions_match(left, right):
    """Incorrect: component label integers are arbitrary."""
    return left.sort_values("vertex")["labels"].equals(
        right.sort_values("vertex")["labels"]
    )
