# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for graph construction and consumption of owning COO containers."""

import pytest

from testing_utils import device_int32

cp = pytest.importorskip("cupy")
cg = pytest.importorskip("cugraph_python")


def test_cycle_consumes_coo():
    ctx = cg.Context()
    coo = cg.make_cycle_coo(ctx, size=100_000)
    graph, properties = cg.create_graph(coo)

    assert coo.consumed
    assert len(properties) == 0
    assert graph.number_of_vertices() == 100_000
    assert graph.number_of_edges() == 100_000


def test_consumed_coo_rejects_second_use():
    ctx = cg.Context()
    coo = cg.make_cycle_coo(ctx, size=8)
    cg.create_graph(coo)

    with pytest.raises(RuntimeError):
        cg.create_graph(coo)


def test_copy_from_arrays_leaves_inputs_valid():
    ctx = cg.Context()
    src = device_int32([0, 1, 2])
    dst = device_int32([1, 2, 0])

    coo = cg.copy_from_arrays(ctx, src, dst)
    graph, _properties = cg.create_graph(coo)

    assert graph.number_of_edges() == 3
    assert cp.array_equal(src, cp.asarray([0, 1, 2], dtype=cp.int32))
    assert cp.array_equal(dst, cp.asarray([1, 2, 0], dtype=cp.int32))


def test_create_graph_rejects_multi_gpu_without_consuming():
    ctx = cg.Context()
    coo = cg.copy_from_arrays(ctx, device_int32([0]), device_int32([1]))

    with pytest.raises(ValueError):
        cg.create_graph(coo, multi_gpu=True)
    assert not coo.consumed

    graph, _properties = cg.create_graph(coo, store_transposed=True)
    assert coo.consumed
    assert graph.number_of_vertices() == 2
    assert graph.number_of_edges() == 1
