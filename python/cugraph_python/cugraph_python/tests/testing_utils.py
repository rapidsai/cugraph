# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the cugraph_python tests."""

import pytest

cp = pytest.importorskip("cupy")


def device_ints(values, dtype):
    cp.cuda.get_current_stream().synchronize()
    return cp.asarray(values, dtype=dtype)


def device_int32(values):
    return device_ints(values, cp.int32)


def device_int64(values):
    return device_ints(values, cp.int64)


def device_float32(values):
    cp.cuda.get_current_stream().synchronize()
    return cp.asarray(values, dtype=cp.float32)


def device_float64(values):
    cp.cuda.get_current_stream().synchronize()
    return cp.asarray(values, dtype=cp.float64)


def to_host(device_array):
    return cp.asnumpy(cp.from_dlpack(device_array))
