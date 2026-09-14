# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free CPU reference; this does not prove GPU execution."""

from collections import deque


def hop_distances(edges, start, cutoff):
    adjacency = {}
    for source, destination in edges:
        adjacency.setdefault(source, []).append(destination)

    distance = {start: 0}
    queue = deque([start])
    while queue:
        source = queue.popleft()
        if distance[source] == cutoff:
            continue
        for destination in adjacency.get(source, []):
            if destination not in distance:
                distance[destination] = distance[source] + 1
                queue.append(destination)
    return distance


def main():
    edges = [
        ("checkout", "payments"),
        ("payments", "ledger"),
        ("payments", "fraud"),
        ("ledger", "archive"),
    ]
    actual = hop_distances(edges, "checkout", 2)
    assert actual == {"checkout": 0, "payments": 1, "ledger": 2, "fraud": 2}
    assert "archive" not in actual
    assert "catalog" not in actual

    reverse = hop_distances([("payments", "checkout")], "checkout", 2)
    assert reverse == {"checkout": 0}
    print("directed reachability reference passed")


if __name__ == "__main__":
    main()
