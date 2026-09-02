# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types

import pytest
import cupy as cp
import numpy as np
import pandas as pd


def test_experimental_warning_wrapper_for_funcs():
    from pylibcugraph.utilities.api_tools import experimental_warning_wrapper

    def EXPERIMENTAL__func(a, b):
        return a - b

    exp_func = experimental_warning_wrapper(EXPERIMENTAL__func)

    with pytest.warns(PendingDeprecationWarning):
        assert 1 == exp_func(3, 2)


def test_experimental_warning_wrapper_for_classes():
    from pylibcugraph.utilities.api_tools import experimental_warning_wrapper

    class EXPERIMENTAL__klass:
        def __init__(self, a, b):
            self.r = a - b

    exp_klass = experimental_warning_wrapper(EXPERIMENTAL__klass)

    with pytest.warns(PendingDeprecationWarning):
        k = exp_klass(3, 2)
        assert 1 == k.r
        assert isinstance(k, exp_klass)
        assert k.__class__.__name__ == "klass"


def test_experimental_warning_wrapper_for_unsupported_type():
    from pylibcugraph.utilities.api_tools import experimental_warning_wrapper

    # A module type should not be allowed to be wrapped
    mod = types.ModuleType("modname")
    with pytest.raises(TypeError):
        experimental_warning_wrapper(mod)


def test_dlpack_memory_accessibility():
    from pylibcugraph.utils import is_device_accessible, is_host_accessible

    host = np.arange(3)
    device = cp.arange(3)
    assert not is_device_accessible(host)
    assert is_host_accessible(host)
    assert is_device_accessible(device)
    assert not is_host_accessible(device)


def test_readonly_pandas_dlpack_memory_accessibility():
    from pylibcugraph.utils import (
        get_c_type_from_py_obj,
        is_device_accessible,
        is_host_accessible,
    )

    series = pd.Series([1, 2, 3])
    assert not series.values.flags.writeable
    assert not is_device_accessible(series)
    assert is_host_accessible(series)
    assert get_c_type_from_py_obj(series) == get_c_type_from_py_obj(np.array([1, 2, 3]))


def test_legacy_dlpack_producer_memory_accessibility():
    from pylibcugraph.utils import is_device_accessible, is_host_accessible

    class LegacyDLPackProducer:
        def __init__(self, array):
            self.array = array

        def __dlpack__(self):
            return self.array.__dlpack__()

    legacy = LegacyDLPackProducer(np.arange(3))
    assert not is_device_accessible(legacy)
    assert is_host_accessible(legacy)
