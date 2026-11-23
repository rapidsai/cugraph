# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for breadth-first search."""

import numpy as np
import pytest
from testing_utils import device_float64, device_int32, device_int64, to_host

cp = pytest.importorskip("cupy")
cg = pytest.importorskip("cugraph_python")

INT32_MAX = np.iinfo(np.int32).max


def test_bfs_small_directed_graph():
    ctx = cg.Context()
    src = device_int32([10, 20, 10])
    dst = device_int32([20, 30, 40])
    graph, _properties = cg.create_graph(cg.copy_from_arrays(ctx, src, dst))

    starts = cg.prepare_vertices(graph, [10])
    vertices, distances, predecessors = cg.bfs(graph, starts, compute_predecessors=True)

    vertices = to_host(vertices)
    distances = to_host(distances)
    predecessors = to_host(predecessors)
    got = {
        int(v): (int(d), int(p)) for v, d, p in zip(vertices, distances, predecessors)
    }

    assert got == {10: (0, -1), 20: (1, 10), 30: (2, 20), 40: (1, 10)}


def test_bfs_directed_edges_limit_reachability():
    ctx = cg.Context()
    src = device_int32([10, 20, 10])
    dst = device_int32([20, 30, 40])
    graph, _properties = cg.create_graph(cg.copy_from_arrays(ctx, src, dst))

    starts = cg.prepare_vertices(graph, [30])
    vertices, distances, predecessors = cg.bfs(graph, starts)

    vertices = to_host(vertices)
    distances = to_host(distances)
    got = {int(v): int(d) for v, d in zip(vertices, distances)}

    assert got == {10: INT32_MAX, 20: INT32_MAX, 30: 0, 40: INT32_MAX}
    assert predecessors is None


def test_result_outlives_graph_and_context():
    ctx = cg.Context()
    src = device_int32([10, 20])
    dst = device_int32([20, 30])
    graph, _properties = cg.create_graph(cg.copy_from_arrays(ctx, src, dst))
    vertices, distances, predecessors = cg.bfs(graph, cg.prepare_vertices(graph, [10]))

    vertices_array = cp.from_dlpack(vertices)
    del vertices, distances, predecessors, graph, ctx

    assert sorted(cp.asnumpy(vertices_array).tolist()) == [10, 20, 30]


def test_bfs_rejects_starts_from_another_graph():
    ctx = cg.Context()
    g1, _p1 = cg.create_graph(
        cg.copy_from_arrays(ctx, device_int32([10]), device_int32([20]))
    )
    g2, _p2 = cg.create_graph(
        cg.copy_from_arrays(ctx, device_int32([10]), device_int32([20]))
    )

    starts = cg.prepare_vertices(g1, [10])

    with pytest.raises(ValueError):
        cg.bfs(g2, starts)


def test_int64_graph_bfs_and_dtype():
    ctx = cg.Context()
    src = device_int64([10, 20, 10])
    dst = device_int64([20, 30, 40])
    graph, _properties = cg.create_graph(cg.copy_from_arrays(ctx, src, dst))
    vertices, distances, predecessors = cg.bfs(
        graph, cg.prepare_vertices(graph, [10]), compute_predecessors=True
    )

    vertices = cp.from_dlpack(vertices)
    distances = cp.from_dlpack(distances)
    predecessors = cp.from_dlpack(predecessors)
    assert vertices.dtype == cp.int64
    assert distances.dtype == cp.int64
    assert predecessors.dtype == cp.int64

    got = {
        int(v): (int(d), int(p))
        for v, d, p in zip(
            cp.asnumpy(vertices), cp.asnumpy(distances), cp.asnumpy(predecessors)
        )
    }
    assert got == {10: (0, -1), 20: (1, 10), 30: (2, 20), 40: (1, 10)}


def test_int64_unreached_distance_is_int64_max():
    ctx = cg.Context()
    graph, _properties = cg.create_graph(
        cg.copy_from_arrays(ctx, device_int64([10, 20, 10]), device_int64([20, 30, 40]))
    )
    vertices, distances, predecessors = cg.bfs(graph, cg.prepare_vertices(graph, [30]))
    assert predecessors is None
    vertices = to_host(vertices)
    distances = to_host(distances)
    got = {int(v): int(d) for v, d in zip(vertices, distances)}
    assert got[30] == 0
    assert got[10] == np.iinfo(np.int64).max


def test_bfs_rejects_transposed_graph():
    ctx = cg.Context()
    plain, _plain_properties = cg.create_graph(
        cg.copy_from_arrays(ctx, device_int32([10, 20]), device_int32([20, 10]))
    )
    transposed, _transposed_properties = cg.create_graph(
        cg.copy_from_arrays(ctx, device_int32([10, 20]), device_int32([20, 10])),
        store_transposed=True,
    )
    starts = cg.prepare_vertices(plain, [10])

    with pytest.raises(ValueError, match="store_transposed"):
        cg.prepare_vertices(transposed, [10])
    with pytest.raises(ValueError, match="store_transposed"):
        cg.bfs(transposed, starts)


def test_bfs_on_a_weighted_graph():
    ctx = cg.Context()
    src = device_int32([10, 20])
    dst = device_int32([20, 30])
    graph, properties = cg.create_graph(
        cg.copy_from_arrays(ctx, src, dst, device_float64([2.0, 4.0]))
    )
    assert len(properties) == 1
    vertices, distances, predecessors = cg.bfs(graph, cg.prepare_vertices(graph, [10]))

    got = {int(v): int(d) for v, d in zip(to_host(vertices), to_host(distances))}
    assert predecessors is None
    assert got == {10: 0, 20: 1, 30: 2}


def test_bfs_rejects_starts_from_a_different_vertex_type():
    ctx = cg.Context()
    g32, _p32 = cg.create_graph(
        cg.copy_from_arrays(ctx, device_int32([10]), device_int32([20]))
    )
    g64, _p64 = cg.create_graph(
        cg.copy_from_arrays(ctx, device_int64([10]), device_int64([20]))
    )
    starts = cg.prepare_vertices(g64, [10])

    with pytest.raises(ValueError):
        cg.bfs(g32, starts)
