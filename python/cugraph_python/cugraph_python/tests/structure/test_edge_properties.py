# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for edge properties returned by graph construction."""

import pytest

from testing_utils import device_float32, device_int32

cp = pytest.importorskip("cupy")
cg = pytest.importorskip("cugraph_python")


def test_copy_from_arrays_rejects_bad_weights():
    ctx = cg.Context()
    src = device_int32([10, 20])
    dst = device_int32([20, 10])

    with pytest.raises(ValueError):
        cg.copy_from_arrays(ctx, src, dst, device_float32([1.0]))
    with pytest.raises(ValueError):
        cg.copy_from_arrays(ctx, src, dst, device_int32([1, 1]))


def test_empty_edge_properties_reject_indexing():
    ctx = cg.Context()
    _graph, properties = cg.create_graph(
        cg.copy_from_arrays(ctx, device_int32([10]), device_int32([20]))
    )

    assert len(properties) == 0
    with pytest.raises(IndexError):
        properties[0]


def test_edge_property_keeps_context_alive():
    ctx = cg.Context()
    graph, properties = cg.create_graph(
        cg.copy_from_arrays(
            ctx,
            device_int32([10, 20, 30]),
            device_int32([20, 30, 10]),
            device_float32([1.0, 1.0, 1.0]),
        ),
        store_transposed=True,
    )
    weights = properties[0]
    del properties, ctx

    _vertices, _scores, _iterations, converged = cg.pagerank(graph, weights)
    assert converged
