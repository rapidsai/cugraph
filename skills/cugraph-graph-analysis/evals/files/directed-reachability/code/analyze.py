# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Starter cuGraph analysis with intentionally incorrect graph semantics."""

import cudf
import cugraph


def reachable_within_hops(edges, vertices, start, max_hops):
    """Return reachable services and hop distance from ``start``."""
    # BUGS: Graph defaults to undirected, isolates are omitted, and SSSP treats
    # larger trust as a cost even though this task asks only for hop reachability.
    graph = cugraph.Graph()
    graph.from_cudf_edgelist(
        edges,
        source="src_service",
        destination="dst_service",
        edge_attr="trust",
    )
    return cugraph.sssp(graph, source=start)


def demo():
    edges = cudf.DataFrame(
        {
            "src_service": ["checkout", "payments", "payments", "ledger"],
            "dst_service": ["payments", "ledger", "fraud", "archive"],
            "trust": [0.99, 0.95, 0.80, 0.90],
        }
    )
    vertices = cudf.Series(
        ["checkout", "payments", "ledger", "fraud", "archive", "catalog"]
    )
    return reachable_within_hops(edges, vertices, "checkout", 2)


if __name__ == "__main__":
    print(demo())
