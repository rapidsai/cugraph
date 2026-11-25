# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PageRank."""

import numpy as np
import pytest

from testing_utils import (
    device_float32,
    device_float64,
    device_int32,
    device_int64,
    to_host,
)

cp = pytest.importorskip("cupy")
cg = pytest.importorskip("cugraph_python")


def _score_map(vertices, pageranks):
    scores = cp.from_dlpack(pageranks)
    return scores.dtype, {
        int(v): float(s) for v, s in zip(to_host(vertices), cp.asnumpy(scores))
    }


def test_pagerank_cycle_is_uniform_and_unrenumbered():
    ctx = cg.Context()
    src = device_int32([10, 20, 30])
    dst = device_int32([20, 30, 10])
    graph, properties = cg.create_graph(
        cg.copy_from_arrays(ctx, src, dst), store_transposed=True
    )

    assert len(properties) == 0
    vertices, pageranks, iterations, converged = cg.pagerank(graph)

    vertices = to_host(vertices)
    scores = cp.from_dlpack(pageranks)
    assert scores.dtype == cp.float32
    scores = cp.asnumpy(scores)

    assert converged
    assert iterations >= 1
    assert sorted(int(v) for v in vertices) == [10, 20, 30]
    assert abs(float(scores.sum()) - 1.0) < 1e-5
    assert np.allclose(scores, 1.0 / 3.0, atol=1e-5)


def test_pagerank_int64_vertices_stay_int64():
    ctx = cg.Context()
    src = device_int64([10, 20, 30])
    dst = device_int64([20, 30, 10])
    graph, _properties = cg.create_graph(
        cg.copy_from_arrays(ctx, src, dst), store_transposed=True
    )

    vertices, pageranks, iterations, converged = cg.pagerank(graph)
    vertices = cp.from_dlpack(vertices)
    scores = cp.from_dlpack(pageranks)

    assert vertices.dtype == cp.int64
    assert scores.dtype == cp.float32
    assert converged
    assert iterations >= 1
    assert sorted(int(v) for v in cp.asnumpy(vertices)) == [10, 20, 30]
    assert abs(float(cp.asnumpy(scores).sum()) - 1.0) < 1e-5


def test_pagerank_rejects_non_transposed_graph():
    ctx = cg.Context()
    graph, _properties = cg.create_graph(
        cg.copy_from_arrays(ctx, device_int32([10, 20]), device_int32([20, 10]))
    )

    with pytest.raises(ValueError, match="store_transposed"):
        cg.pagerank(graph)


def test_weighted_pagerank_follows_float32_weights():
    ctx = cg.Context()
    src = device_int32([10, 10, 20, 30])
    dst = device_int32([20, 30, 10, 10])
    weights = device_float32([1.0, 9.0, 1.0, 1.0])
    graph, properties = cg.create_graph(
        cg.copy_from_arrays(ctx, src, dst, weights), store_transposed=True
    )

    assert len(properties) == 1
    vertices, pageranks, iterations, converged = cg.pagerank(graph, properties[0])
    dtype, got = _score_map(vertices, pageranks)

    assert converged
    assert iterations >= 1
    assert dtype == cp.float32
    assert got[30] > got[20] * 2
    assert abs(sum(got.values()) - 1.0) < 1e-4
    assert float(weights[1]) == 9.0


def test_weighted_pagerank_float64_weights_and_int64_vertices():
    ctx = cg.Context()
    src = device_int64([10, 10, 20, 30])
    dst = device_int64([20, 30, 10, 10])
    weights = device_float64([1.0, 9.0, 1.0, 1.0])
    graph, properties = cg.create_graph(
        cg.copy_from_arrays(ctx, src, dst, weights), store_transposed=True
    )

    assert len(properties) == 1
    vertices, pageranks, iterations, converged = cg.pagerank(graph, properties[0])
    vertices = to_host(vertices)
    scores = cp.from_dlpack(pageranks)
    got = {int(v): float(s) for v, s in zip(vertices, cp.asnumpy(scores))}

    assert vertices.dtype == np.int64
    assert scores.dtype == cp.float64
    assert converged
    assert iterations >= 1
    assert got[30] > got[20] * 2
    assert abs(sum(got.values()) - 1.0) < 1e-8


def test_unweighted_symmetric_targets_have_equal_pagerank():
    ctx = cg.Context()
    src = device_int32([10, 10, 20, 30])
    dst = device_int32([20, 30, 10, 10])
    graph, _properties = cg.create_graph(
        cg.copy_from_arrays(ctx, src, dst), store_transposed=True
    )

    vertices, pageranks, iterations, converged = cg.pagerank(graph)
    _, got = _score_map(vertices, pageranks)

    assert converged
    assert abs(got[30] - got[20]) < 1e-5


def test_pagerank_rejects_property_from_another_graph():
    ctx = cg.Context()
    sources = device_int32([10, 20, 30])
    destinations = device_int32([20, 30, 10])
    weights = device_float32([1.0, 1.0, 1.0])
    graph, _properties = cg.create_graph(
        cg.copy_from_arrays(ctx, sources, destinations, weights), store_transposed=True
    )
    _other, other_properties = cg.create_graph(
        cg.copy_from_arrays(
            ctx, device_int32([10, 20, 30]), device_int32([20, 30, 10]), weights
        ),
        store_transposed=True,
    )

    with pytest.raises(ValueError, match="different graph"):
        cg.pagerank(graph, other_properties[0])


def test_pagerank_rejects_property_of_a_different_edge_type():
    ctx = cg.Context()
    graph, _properties = cg.create_graph(
        cg.copy_from_arrays(
            ctx,
            device_int32([10, 20, 30]),
            device_int32([20, 30, 10]),
            device_float32([1, 1, 1]),
        ),
        store_transposed=True,
    )
    _other, int64_properties = cg.create_graph(
        cg.copy_from_arrays(
            ctx,
            device_int64([10, 20, 30]),
            device_int64([20, 30, 10]),
            device_float64([1, 1, 1]),
        ),
        store_transposed=True,
    )

    with pytest.raises(ValueError, match="does not match"):
        cg.pagerank(graph, int64_properties[0])
