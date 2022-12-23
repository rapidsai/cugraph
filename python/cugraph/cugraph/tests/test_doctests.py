# Copyright (c) 2022, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import doctest
import inspect
import io
import os

import numpy as np
import pandas as pd
import scipy
import pytest
import re

import cugraph
import pylibcugraph
import cudf
from numba import cuda
from cugraph.testing import utils


modules_to_skip = ["dask", "proto", "raft"]
datasets = utils.RAPIDS_DATASET_ROOT_DIR_PATH

cuda_version_string = ".".join([str(n) for n in cuda.runtime.get_version()])


def _is_public_name(name):
    return not name.startswith("_")


def _is_python_module(member):
    return os.path.splitext(member.__file__)[1] == ".py"


def _module_from_library(member, libname):
    return libname in member.__module__


def _file_from_library(member, libname):
    return libname in member.__file__


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
        # FIXME: when PropertyGraph is removed from EXPERIMENTAL
        # manually including PropertyGraph until it is removed from EXPERIMENTAL
        is_pg = "PropertyGraph" in str(docstring)
        is_experimental = "EXPERIMENTAL" in str(docstring) and not is_pg
        # if has_examples and not is_dask:
        if has_examples and not is_dask and not is_experimental:
            yield docstring


def _fetch_doctests():
    finder = doctest.DocTestFinder()
    yield from _find_modules_in_obj(finder, cugraph, "cugraph", _is_public_name)
    yield from _find_modules_in_obj(
        finder, pylibcugraph, "pylibcugraph", _is_public_name
    )


def skip_docstring(docstring):
    # Depending on different builds or architectures, some examples
    # won't work.
    first_line = docstring.examples[0].source

    if (
        re.search("does not run on CUDA", first_line)
        and cuda_version_string in first_line
    ):
        return True
    return False


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

    @pytest.mark.parametrize(
        "docstring", _fetch_doctests(), ids=lambda docstring: docstring.name
    )
    def test_docstring(self, docstring):
        # We ignore differences in whitespace in the doctest output, and enable
        # the use of an ellipsis "..." to match any string in the doctest
        # output. An ellipsis is useful for, e.g., memory addresses or
        # imprecise floating point values.
        if skip_docstring(docstring):
            print("Skipped!")
            return

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
            # If some failed but all the failures were due to lack of
            # cugraph-ops support, we can skip.
            out = doctest_stdout.getvalue()
            if ("CUGRAPH_UNKNOWN_ERROR" in out and "unimplemented" in out) or (
                "built with NO_CUGRAPH_OPS" in out
            ):
                pytest.skip("Doctest requires cugraph-ops support.")
            raise
