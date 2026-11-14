# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf
import cugraph
import cugraph.dask as dcg
import dask_cudf
import numpy as np
import pytest
from cugraph.tests.components.test_simple_cycles import (
    _assert_cycles,
    _edgelist,
    _expected,
)


@pytest.mark.mg
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("bound", [1, 2, 3, 4])
@pytest.mark.parametrize("seeds", [None, [], [3], [3, 3, 4]])
@pytest.mark.parametrize("distributed_seeds", [False, True])
def test_simple_cycles_mg(dask_client, dtype, bound, seeds, distributed_seeds):
    graph = cugraph.Graph(directed=True)
    edges = dask_cudf.from_cudf(_edgelist(dtype), npartitions=2)
    graph.from_dask_cudf_edgelist(edges, source="src", destination="dst")
    seed_input = seeds
    if distributed_seeds and seeds is not None:
        # A single partition exercises ranks with no seed partitions, which
        # must still participate using a non-NULL empty array.
        seed_input = dask_cudf.from_cudf(cudf.Series(seeds, dtype=dtype), npartitions=1)
    result = dcg.simple_cycles(graph, bound, seed_input).compute()
    _assert_cycles(result, _expected(bound, seeds))


@pytest.mark.mg
@pytest.mark.parametrize("seeds", [None, [], [3]])
def test_simple_cycles_mg_string_vertices(dask_client, seeds):
    labels = [f"node-{v}" for v in range(5)]
    graph = cugraph.Graph(directed=True)
    graph.from_dask_cudf_edgelist(
        dask_cudf.from_cudf(_edgelist(None, labels), npartitions=2),
        source="src",
        destination="dst",
    )
    seed_input = None if seeds is None else [labels[v] for v in seeds]
    result = dcg.simple_cycles(graph, 4, seed_input).compute()
    _assert_cycles(result, _expected(4, seeds, labels))


@pytest.mark.mg
def test_simple_cycles_mg_multicolumn_vertices(dask_client):
    edges = _edgelist(np.int32)
    edges["src_name"] = edges.src.astype(str)
    edges["dst_name"] = edges.dst.astype(str)
    graph = cugraph.Graph(directed=True)
    graph.from_dask_cudf_edgelist(
        dask_cudf.from_cudf(edges, npartitions=2),
        source=["src", "src_name"],
        destination=["dst", "dst_name"],
    )
    seeds = dask_cudf.from_cudf(
        cudf.DataFrame({"id": [3], "name": ["3"]}), npartitions=1
    )
    result = dcg.simple_cycles(graph, 4, seeds).compute()
    assert (result["0_vertex"].astype(str) == result["1_vertex"]).all()
    _assert_cycles(result.rename(columns={"0_vertex": "vertex"}), _expected(4, [3]))


@pytest.mark.mg
def test_simple_cycles_mg_acyclic(dask_client):
    graph = cugraph.Graph(directed=True)
    graph.from_dask_cudf_edgelist(
        dask_cudf.from_cudf(
            cudf.DataFrame({"src": [0, 1], "dst": [1, 2]}), npartitions=2
        ),
        source="src",
        destination="dst",
    )
    result = dcg.simple_cycles(graph, 4).compute()
    assert list(result.columns) == ["cycle_id", "position", "vertex"]
    assert len(result) == 0


@pytest.mark.mg
@pytest.mark.parametrize(
    "bound,error",
    [(0, ValueError), (-1, ValueError), (True, TypeError), (2.5, TypeError)],
)
def test_simple_cycles_mg_invalid_bound(dask_client, bound, error):
    graph = cugraph.Graph(directed=True)
    graph.from_dask_cudf_edgelist(
        dask_cudf.from_cudf(_edgelist(np.int32), npartitions=2),
        source="src",
        destination="dst",
    )
    with pytest.raises(error, match="length_bound"):
        dcg.simple_cycles(graph, bound)


@pytest.mark.mg
@pytest.mark.parametrize("seed_input", [3, [None]])
def test_simple_cycles_mg_invalid_seeds(dask_client, seed_input):
    graph = cugraph.Graph(directed=True)
    graph.from_dask_cudf_edgelist(
        dask_cudf.from_cudf(_edgelist(np.int32), npartitions=2),
        source="src",
        destination="dst",
    )
    with pytest.raises((TypeError, ValueError), match="seed_vertices"):
        dcg.simple_cycles(graph, 4, seed_input)


@pytest.mark.mg
def test_simple_cycles_mg_undirected(dask_client):
    graph = cugraph.Graph()
    graph.from_dask_cudf_edgelist(
        dask_cudf.from_cudf(_edgelist(np.int32), npartitions=2),
        source="src",
        destination="dst",
    )
    with pytest.raises(ValueError, match="directed"):
        dcg.simple_cycles(graph, 4)
