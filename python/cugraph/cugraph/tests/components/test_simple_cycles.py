# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf
import cugraph
import networkx as nx
import numpy as np
import pytest

SRC = [0, 1, 1, 2, 2, 3, 4]
DST = [1, 2, 3, 0, 1, 2, 4]


def _canonical(cycle):
    i = cycle.index(min(cycle))
    return tuple(cycle[i:] + cycle[:i])


def _assert_cycles(result, expected):
    pdf = result.to_pandas().sort_values(["cycle_id", "position"])
    actual = []
    for _, cycle in pdf.groupby("cycle_id"):
        assert cycle.position.tolist() == list(range(len(cycle)))
        vertices = cycle.vertex.tolist()
        assert len(vertices) == len(set(vertices))
        actual.append(_canonical(vertices))
    # Compare lists rather than sets to detect duplicate cycles across GPUs.
    assert sorted(actual) == sorted(_canonical(c) for c in expected)


def _expected(bound, seeds=None, labels=None):
    graph = nx.DiGraph(zip(SRC, DST))
    cycles = [c for c in nx.simple_cycles(graph) if len(c) <= bound]
    if seeds is not None:
        cycles = [c for c in cycles if any(v in seeds for v in c)]
    if labels is not None:
        cycles = [[labels[v] for v in c] for c in cycles]
    return cycles


def _edgelist(dtype, labels=None):
    if labels is None:
        return cudf.DataFrame(
            {
                "src": cudf.Series(SRC, dtype=dtype),
                "dst": cudf.Series(DST, dtype=dtype),
            }
        )
    return cudf.DataFrame(
        {"src": [labels[v] for v in SRC], "dst": [labels[v] for v in DST]}
    )


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("renumber", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("bound", [1, 2, 3, 4])
@pytest.mark.parametrize("seeds", [None, [], [3], [3, 3, 4]])
def test_simple_cycles(dtype, renumber, transposed, bound, seeds):
    graph = cugraph.Graph(directed=True)
    graph.from_cudf_edgelist(
        _edgelist(dtype),
        source="src",
        destination="dst",
        renumber=renumber,
        store_transposed=transposed,
    )
    result = cugraph.simple_cycles(graph, bound, seeds)
    _assert_cycles(result, _expected(bound, seeds))


@pytest.mark.parametrize("seeds", [None, [], [3]])
def test_string_vertices(seeds):
    labels = [f"node-{v}" for v in range(5)]
    graph = cugraph.Graph(directed=True)
    graph.from_cudf_edgelist(_edgelist(None, labels), source="src", destination="dst")
    external_seeds = None if seeds is None else [labels[v] for v in seeds]
    result = cugraph.simple_cycles(graph, 4, external_seeds)
    _assert_cycles(result, _expected(4, seeds, labels))


def test_multicolumn_vertices():
    edges = _edgelist(np.int32)
    edges["src_name"] = edges.src.astype(str)
    edges["dst_name"] = edges.dst.astype(str)
    graph = cugraph.Graph(directed=True)
    graph.from_cudf_edgelist(
        edges, source=["src", "src_name"], destination=["dst", "dst_name"]
    )
    result = cugraph.simple_cycles(graph, 4, cudf.DataFrame({"id": [3], "name": ["3"]}))
    assert (result["0_vertex"].astype(str) == result["1_vertex"]).all()
    _assert_cycles(result.rename(columns={"0_vertex": "vertex"}), _expected(4, [3]))


@pytest.mark.parametrize(
    "bound,error",
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (2.5, TypeError),
        (2**64, ValueError),
    ],
)
def test_invalid_bound(bound, error):
    graph = cugraph.Graph(directed=True)
    graph.from_cudf_edgelist(_edgelist(np.int32), source="src", destination="dst")
    with pytest.raises(error, match="length_bound"):
        cugraph.simple_cycles(graph, bound)


def test_undirected():
    graph = cugraph.Graph()
    graph.from_cudf_edgelist(_edgelist(np.int32), source="src", destination="dst")
    with pytest.raises(ValueError, match="directed"):
        cugraph.simple_cycles(graph, 4)


@pytest.mark.parametrize("seeds,error", [(3, TypeError), ([None], ValueError)])
def test_invalid_seeds(seeds, error):
    graph = cugraph.Graph(directed=True)
    graph.from_cudf_edgelist(_edgelist(np.int32), source="src", destination="dst")
    with pytest.raises(error, match="seed_vertices"):
        cugraph.simple_cycles(graph, 4, seeds)


def test_acyclic():
    graph = cugraph.Graph(directed=True)
    graph.from_cudf_edgelist(
        cudf.DataFrame({"src": [0, 1], "dst": [1, 2]}),
        source="src",
        destination="dst",
    )
    result = cugraph.simple_cycles(graph, 4)
    assert list(result.columns) == ["cycle_id", "position", "vertex"]
    assert len(result) == 0


@pytest.mark.parametrize("seed_type", [cudf.Series, cudf.DataFrame])
def test_seed_containers(seed_type):
    graph = cugraph.Graph(directed=True)
    graph.from_cudf_edgelist(_edgelist(np.int32), source="src", destination="dst")
    seeds = cudf.Series([3], dtype=np.int64)
    if seed_type is cudf.DataFrame:
        seeds = seeds.to_frame(name="seed")
    _assert_cycles(cugraph.simple_cycles(graph, 4, seeds), _expected(4, [3]))


def test_unknown_string_seed():
    graph = cugraph.Graph(directed=True)
    graph.from_cudf_edgelist(
        _edgelist(None, [f"node-{v}" for v in range(5)]),
        source="src",
        destination="dst",
    )
    with pytest.raises(ValueError, match="belong to the graph"):
        cugraph.simple_cycles(graph, 4, ["unknown"])
