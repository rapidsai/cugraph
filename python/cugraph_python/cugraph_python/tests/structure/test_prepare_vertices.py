# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for vertex preparation (external to internal ID translation)."""

import pytest

from testing_utils import device_int32

cp = pytest.importorskip("cupy")
cg = pytest.importorskip("cugraph_python")


def test_prepare_rejects_unknown_vertex():
    ctx = cg.Context()
    graph, _properties = cg.create_graph(
        cg.copy_from_arrays(ctx, device_int32([10]), device_int32([20]))
    )

    with pytest.raises(ValueError):
        cg.prepare_vertices(graph, [999])


def test_prepare_rejects_id_outside_vertex_type():
    ctx = cg.Context()
    graph, _properties = cg.create_graph(
        cg.copy_from_arrays(ctx, device_int32([0]), device_int32([1]))
    )

    with pytest.raises(ValueError):
        cg.prepare_vertices(graph, [2**31])
