# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free component contract checks; no GPU execution occurs."""

import ast
from collections import defaultdict, deque
from pathlib import Path


def weak_partition(edges, vertices):
    adjacency = defaultdict(set)
    for source, destination in edges:
        adjacency[source].add(destination)
        adjacency[destination].add(source)
    unseen = set(vertices)
    groups = []
    while unseen:
        start = min(unseen)
        queue = deque([start])
        group = set()
        while queue:
            vertex = queue.popleft()
            if vertex in group:
                continue
            group.add(vertex)
            unseen.discard(vertex)
            queue.extend(adjacency[vertex] - group)
        groups.append(frozenset(group))
    return frozenset(groups)


def calls(tree, suffix):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == suffix
    ]


def main():
    assert weak_partition(
        [("api", "db"), ("worker", "db"), ("a", "b"), ("b", "a")],
        ["api", "db", "worker", "a", "b", "isolate"],
    ) == frozenset(
        [
            frozenset(["api", "db", "worker"]),
            frozenset(["a", "b"]),
            frozenset(["isolate"]),
        ]
    )

    source = Path(__file__).with_name("analyze_components.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    graph_calls = calls(tree, "Graph")
    directions = {
        next((kw.value.value for kw in call.keywords if kw.arg == "directed" and isinstance(kw.value, ast.Constant)), None)
        for call in graph_calls
    }
    assert directions >= {True, False}, "construct separate directed and undirected graph views"

    builds = calls(tree, "from_cudf_edgelist")
    assert len(builds) >= 2
    for build in builds:
        names = {kw.arg for kw in build.keywords}
        assert "vertices" in names and "renumber" in names

    for function in ("weakly_connected_components", "strongly_connected_components"):
        call = calls(tree, function)[0]
        forbidden = {kw.arg for kw in call.keywords} & {"directed", "connection", "return_labels"}
        assert not forbidden, f"Graph input does not accept these {function} arguments: {forbidden}"

    lowered = source.lower()
    assert "canonical" in lowered or "frozenset" in lowered, "compare partitions independent of label numbers"
    assert "duplicated" in lowered and ("nunique" in lowered or "len(" in lowered), (
        "validate one keyed component row per required vertex"
    )
    print("component source contract passed; GPU execution not exercised")


if __name__ == "__main__":
    main()
