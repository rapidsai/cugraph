# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import types

import pytest


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
