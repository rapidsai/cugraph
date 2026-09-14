# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free structural checks; these do not prove GPU execution."""

import ast
from pathlib import Path


def calls(tree, suffix):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == suffix
    ]


def keyword(call, name):
    return next((item.value for item in call.keywords if item.arg == name), None)


def is_true(node):
    return isinstance(node, ast.Constant) and node.value is True


def main():
    path = Path(__file__).with_name("rank_pages.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    graph_call = calls(tree, "Graph")[0]
    assert is_true(keyword(graph_call, "directed")), "PageRank graph must be directed"

    build = calls(tree, "from_cudf_edgelist")[0]
    assert keyword(build, "vertices") is not None, (
        "complete page universe must be supplied"
    )
    assert is_true(keyword(build, "renumber")), (
        "external page IDs require renumber=True"
    )
    assert is_true(keyword(build, "store_transposed")), (
        "PageRank graph should store the transpose"
    )
    assert keyword(build, "edge_attr") is None and keyword(build, "weight") is None, (
        "latency_ms is not PageRank transition strength; build this graph unweighted"
    )

    pagerank_call = calls(tree, "pagerank")[0]
    for name in ("alpha", "max_iter", "tol", "fail_on_nonconvergence"):
        assert keyword(pagerank_call, name) is not None, (
            f"missing PageRank parameter: {name}"
        )

    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    assert any(
        isinstance(target, (ast.Tuple, ast.List))
        and len(target.elts) == 2
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "pagerank"
        for node in assignments
        for target in node.targets
    ), "fail_on_nonconvergence=False requires tuple unpacking"

    lowered = source.lower()
    assert "converged" in lowered and ("raise" in lowered or "runtimeerror" in lowered)
    assert "isfinite" in lowered, "validate finite PageRank scores"
    assert "nunique" in lowered or "duplicated" in lowered, (
        "validate keyed result coverage"
    )
    assert ".sum(" in lowered, "validate PageRank score mass"
    print("PageRank source contract passed; GPU execution not exercised")


if __name__ == "__main__":
    main()
