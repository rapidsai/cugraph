#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic companion grader for cuGraph analysis evals."""

import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

TRAJECTORY = Path(os.environ.get("HARBOR_ATIF_PATH", "/logs/agent/trajectory.json"))
ENTRY = Path(os.environ.get("HARBOR_ENTRY_JSON", "/tests/entry.json"))
REWARD_JSON = Path(os.environ.get("HARBOR_REWARD_JSON", "/logs/verifier/reward.json"))
REWARD_TXT = Path(os.environ.get("HARBOR_REWARD_TXT", "/logs/verifier/reward.txt"))
WORKSPACE = Path(os.environ.get("HARBOR_WORKSPACE", "/workspace"))


def find_fixture(filename):
    matches = list(WORKSPACE.rglob(filename))
    return matches[0].resolve() if matches else None


def trajectory_text():
    if not TRAJECTORY.exists():
        return ""
    data = json.loads(TRAJECTORY.read_text(encoding="utf-8"))
    return " ".join(
        json.dumps(step.get("message", ""))
        for step in data.get("steps", [])
        if step.get("source") == "agent"
    ).lower()


def ast_calls(tree, name):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


def bool_keyword(call, name, expected):
    value = next((kw.value for kw in call.keywords if kw.arg == name), None)
    return isinstance(value, ast.Constant) and value.value is expected


def main():
    entry = json.loads(ENTRY.read_text())
    case_id = entry["id"]
    text = trajectory_text()
    checks = {}

    if case_id == "cugraph-analysis-explicit-directed-reachability":
        source = find_fixture("analyze.py")
        reference = find_fixture("reference_smoke.py")
        code = source.read_text().lower() if source else ""
        syntax = (
            subprocess.run(
                ["python3", "-m", "py_compile", str(source)], capture_output=True
            ).returncode
            == 0
            if source
            else False
        )
        trusted = bool(
            reference
            and hashlib.sha256(reference.read_bytes()).hexdigest()
            == "e70cdab22873c0d14365dcffdd8d1457f2289075c777dd3a6b2cffe08e6aee98"
        )
        ref_ok = (
            subprocess.run(["python3", str(reference)], capture_output=True).returncode
            == 0
            if trusted
            else False
        )
        checks = {
            "syntax_and_reference": (0.2, syntax and ref_ok),
            "directed_graph": (0.25, "directed=true" in code.replace(" ", "")),
            "bfs_hops": (0.3, "bfs" in code and "sssp" not in code),
            "depth_and_keys": (0.15, "depth_limit" in code and "service" in code),
            "execution_evidence": (
                0.1,
                "cpu" in text and ("gpu" in text or "cugraph" in text),
            ),
        }
    elif case_id == "cugraph-analysis-explicit-pagerank-contract":
        source = find_fixture("rank_pages.py")
        code = source.read_text(encoding="utf-8") if source else ""
        lowered = code.lower()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            tree = ast.parse("")
        graph_calls = ast_calls(tree, "Graph")
        builds = ast_calls(tree, "from_cudf_edgelist")
        rank_calls = ast_calls(tree, "pagerank")
        graph_ok = bool(graph_calls and bool_keyword(graph_calls[0], "directed", True))
        build_ok = bool(
            builds
            and {kw.arg for kw in builds[0].keywords}
            >= {"vertices", "renumber", "store_transposed"}
            and bool_keyword(builds[0], "renumber", True)
            and bool_keyword(builds[0], "store_transposed", True)
        )
        unweighted = bool(
            builds
            and not ({kw.arg for kw in builds[0].keywords} & {"edge_attr", "weight"})
        )
        tuple_unpack = any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == 2
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "pagerank"
            for node in ast.walk(tree)
        )
        convergence = bool(
            rank_calls
            and bool_keyword(rank_calls[0], "fail_on_nonconvergence", False)
            and tuple_unpack
            and "converged" in lowered
        )
        finite_check = "isfinite" in lowered or (
            "isna" in lowered and "scores <" in lowered and "scores >" in lowered
        )
        validation = (
            finite_check
            and ("nunique" in lowered or "duplicated" in lowered)
            and ".sum(" in lowered
        )
        syntax_ok = (
            subprocess.run(
                ["python3", "-m", "py_compile", str(source)], capture_output=True
            ).returncode
            == 0
            if source
            else False
        )
        checks = {
            "syntax": (0.1, syntax_ok),
            "graph_contract": (0.25, graph_ok and build_ok),
            "unweighted_semantics": (0.1, unweighted),
            "convergence_contract": (0.2, convergence),
            "result_validation": (0.25, validation),
            "execution_evidence": (
                0.1,
                "gpu" in text
                and any(
                    term in text for term in ("not", "block", "unavailable", "separate")
                ),
            ),
        }
    elif case_id == "cugraph-analysis-explicit-component-semantics":
        source = find_fixture("analyze_components.py")
        code = source.read_text(encoding="utf-8") if source else ""
        lowered = code.lower()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            tree = ast.parse("")
        graph_calls = ast_calls(tree, "Graph")
        builds = ast_calls(tree, "from_cudf_edgelist")
        directions = {
            expected
            for expected in (True, False)
            if any(bool_keyword(call, "directed", expected) for call in graph_calls)
        }
        builds_ok = len(builds) >= 2 and all(
            {kw.arg for kw in call.keywords} >= {"vertices", "renumber"}
            and bool_keyword(call, "renumber", True)
            for call in builds[:2]
        )
        component_calls = ast_calls(tree, "weakly_connected_components") + ast_calls(
            tree, "strongly_connected_components"
        )
        api_ok = len(component_calls) >= 2 and all(
            not (
                {kw.arg for kw in call.keywords}
                & {"directed", "connection", "return_labels"}
            )
            for call in component_calls
        )
        partition_ok = (
            "canonical" in lowered or "frozenset" in lowered
        ) and "duplicated" in lowered
        syntax_ok = (
            subprocess.run(
                ["python3", "-m", "py_compile", str(source)], capture_output=True
            ).returncode
            == 0
            if source
            else False
        )
        checks = {
            "syntax": (0.1, syntax_ok),
            "separate_graph_views": (0.2, directions == {True, False}),
            "complete_keyed_builds": (0.2, builds_ok),
            "component_api_contract": (0.2, api_ok),
            "partition_validation": (0.2, partition_ok),
            "execution_evidence": (
                0.1,
                "gpu" in text
                and any(
                    term in text for term in ("not", "block", "unavailable", "separate")
                ),
            ),
        }
    elif case_id == "cugraph-analysis-implicit-parallel-edges":
        checks = {
            "graph_vs_multigraph": (
                0.25,
                "simple graph" in text and "multigraph" in text,
            ),
            "aggregation_is_semantic": (
                0.25,
                "aggregate" in text
                and any(
                    term in text
                    for term in ("semantic", "meaning", "question", "choose")
                ),
            ),
            "algorithm_support": (
                0.25,
                "algorithm" in text
                and any(term in text for term in ("support", "compatible", "accept")),
            ),
            "edge_reconciliation": (
                0.25,
                any(
                    term in text
                    for term in (
                        "reconcil",
                        "raw edge",
                        "edge count",
                        "transfer count",
                        "number_of_edges",
                        "len(raw)",
                    )
                ),
            ),
        }
    elif case_id == "cugraph-analysis-contextual-centrality":
        checks = {
            "distinct_meanings": (
                0.25,
                "in-degree" in text and "pagerank" in text and "betweenness" in text,
            ),
            "no_unexplained_average": (0.25, "average" in text and "not" in text),
            "parameters_and_convergence": (
                0.25,
                "alpha" in text
                and ("tolerance" in text or "tol" in text)
                and "converg" in text,
            ),
            "graph_scoped_interpretation": (
                0.25,
                "graph" in text
                and any(
                    term in text
                    for term in ("within", "definition", "scope", "does not")
                ),
            ),
        }
    elif case_id == "cugraph-analysis-implicit-composite-identities":
        checks = {
            "rejects_concatenation": (
                0.25,
                "concat" in text
                and any(
                    term in text for term in ("collision", "not", "avoid", "do not")
                ),
            ),
            "column_lists": (
                0.25,
                any(
                    term in text.replace(" ", "") for term in ("source=[", "source=src")
                )
                and any(
                    term in text.replace(" ", "")
                    for term in ("destination=[", "destination=dst")
                ),
            ),
            "renumbers": (0.25, "renumber=true" in text.replace(" ", "")),
            "keyed_handoff": (
                0.25,
                "0_vertex" in text
                and "1_vertex" in text
                and ("tenant" in text or "account" in text),
            ),
        }
    elif case_id == "cugraph-analysis-contextual-symmetrization-metadata":
        checks = {
            "diagnoses_incompatibility": (
                0.25,
                "symmetr" in text
                and "edge_id" in text
                and "edge_type" in text
                and any(
                    term in text
                    for term in ("incompat", "cannot", "not supported", "fails")
                ),
            ),
            "separates_artifacts": (
                0.25,
                any(
                    term in text
                    for term in (
                        "raw edge",
                        "source edge",
                        "typed edge",
                        "raw relationship",
                        "relationship_lineage",
                        "audit_edges",
                        "audit sidecar",
                    )
                )
                and "topology" in text,
            ),
            "reconciles_topology": (
                0.25,
                any(
                    term in text
                    for term in (
                        "reconcil",
                        "edge count",
                        "distinct pair",
                        "relationship count",
                        "distinct-topology",
                        "row counts",
                    )
                ),
            ),
            "no_invented_identity": (
                0.25,
                "reverse" in text
                and any(
                    term in text for term in ("do not", "don't", "not invent", "policy")
                ),
            ),
        }
    else:
        checks = {
            "negative_routing": (0.7, "cudf" in text or "tabular" in text),
            "no_graph": (
                0.3,
                "do not" in text or "unnecessary" in text or "no graph" in text,
            ),
        }

    score = sum(weight for weight, passed in checks.values() if passed)
    details = {
        name: {"score": float(passed), "reason": "passed" if passed else "failed"}
        for name, (_, passed) in checks.items()
    }
    reward = {
        "overall": score,
        "custom_metrics": {"graph_contract_correctness": score},
        "details": details,
    }
    REWARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    REWARD_JSON.write_text(json.dumps(reward, indent=2))
    REWARD_TXT.write_text(str(score))


if __name__ == "__main__":
    main()
