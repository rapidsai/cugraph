# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import doctest
import inspect
import io
import os

import numpy as np
import pandas as pd
import scipy
import pytest

import cugraph
import pylibcugraph
import cudf
from cugraph.testing import utils

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

modules_to_skip = ["dask", "proto", "raft"]
datasets = utils.RAPIDS_DATASET_ROOT_DIR_PATH


def _is_public_name(name):
    return not name.startswith("_")


def _is_python_module(member):
    member_file = getattr(member, "__file__", "")
    return os.path.splitext(member_file)[1] == ".py"


def _module_from_library(member, libname):
    return libname in getattr(member, "__file__", "")


def _file_from_library(member, libname):
    return libname in getattr(member, "__file__", "")


def _find_modules_in_obj(finder, obj, obj_name, criteria=None):
    for name, member in inspect.getmembers(obj):
        if criteria is not None and not criteria(name):
            continue
        if inspect.ismodule(member) and (member not in modules_to_skip):
            yield from _find_doctests_in_obj(finder, member, obj_name, _is_public_name)


def _find_doctests_in_obj(finder, obj, obj_name, criteria=None):
    """Find all doctests in a module or class.
    Parameters
    ----------
    finder : doctest.DocTestFinder
        The DocTestFinder object to use.

    obj : module or class
        The object to search for docstring examples.

    criteria : callable, optional

    Yields
    ------
    doctest.DocTest
        The next doctest found in the object.
    """
    for name, member in inspect.getmembers(obj):
        if criteria is not None and not criteria(name):
            continue

        if inspect.ismodule(member):
            if _file_from_library(member, obj_name) and _is_python_module(member):
                _find_doctests_in_obj(finder, member, obj_name, criteria)
        if inspect.isfunction(member):
            yield from _find_doctests_in_docstring(finder, member)
        if inspect.isclass(member):
            if member.__module__ and _module_from_library(member, obj_name):
                yield from _find_doctests_in_docstring(finder, member)


def _find_doctests_in_docstring(finder, member):
    for docstring in finder.find(member):
        has_examples = docstring.examples
        is_dask = "dask" in str(docstring)
        is_experimental = "EXPERIMENTAL" in str(docstring)
        # if has_examples and not is_dask:
        if has_examples and not is_dask and not is_experimental:
            yield docstring


def _fetch_doctests():
    finder = doctest.DocTestFinder()
    yield from _find_modules_in_obj(finder, cugraph, "cugraph", _is_public_name)
    yield from _find_modules_in_obj(
        finder, pylibcugraph, "pylibcugraph", _is_public_name
    )


def skip_docstring(docstring_obj):
    """
    Returns a string indicating why the doctest example string should not be
    tested, or None if it should be tested.  This string can be used as the
    "reason" arg to pytest.skip().
    """
    docstring = docstring_obj.docstring
    for line in docstring.splitlines():
        if "random_walks" in line:
            return (
                "docstring example not supported for random walks"
                "because of the random nature of the results"
            )
    return None


class TestDoctests:
    abs_datasets_path = datasets.absolute()

    @pytest.fixture(autouse=True)
    def chdir_to_tmp_path(cls, tmp_path):
        original_directory = os.getcwd()
        try:
            os.chdir(tmp_path)
            yield
        finally:
            os.chdir(original_directory)

    @pytest.mark.sg
    @pytest.mark.parametrize(
        "docstring", _fetch_doctests(), ids=lambda docstring: docstring.name
    )
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_docstring(self, docstring):
        # We ignore differences in whitespace in the doctest output, and enable
        # the use of an ellipsis "..." to match any string in the doctest
        # output. An ellipsis is useful for, e.g., memory addresses or
        # imprecise floating point values.
        skip_reason = skip_docstring(docstring)
        if skip_reason is not None:
            pytest.skip(reason=skip_reason)

        optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        runner = doctest.DocTestRunner(optionflags=optionflags)
        np.random.seed(6)
        globs = dict(
            cudf=cudf,
            np=np,
            cugraph=cugraph,
            datasets_path=self.abs_datasets_path,
            scipy=scipy,
            pd=pd,
        )
        docstring.globs = globs

        # Capture stdout and include failing outputs in the traceback.
        doctest_stdout = io.StringIO()
        with contextlib.redirect_stdout(doctest_stdout):
            runner.run(docstring)
            results = runner.summarize()
        try:
            assert not results.failed, (
                f"{results.failed} of {results.attempted} doctests failed for "
                f"{docstring.name}:\n{doctest_stdout.getvalue()}"
            )
        except AssertionError:
            raise
