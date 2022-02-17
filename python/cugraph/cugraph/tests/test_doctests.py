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
import pathlib

import numpy as np
import pandas as pd
import scipy
import pytest

import cugraph
import cudf
from numba import cuda


modules_to_skip = ["dask", "proto", "raft"]
datasets = pathlib.Path(cugraph.__path__[0]).parent.parent.parent / "datasets"


def _is_public_name(name):
    return not name.startswith("_")


def _is_python_module(member):
    return os.path.splitext(member.__file__)[1] == '.py'


def _module_from_cugraph(member):
    return 'cugraph' in member.__module__


def _file_from_cugraph(member):
    return 'cugraph' in member.__file__


def _find_modules_in_obj(finder, obj, criteria=None):
    for name, member in inspect.getmembers(obj):
        if criteria is not None and not criteria(name):
            continue
        if inspect.ismodule(member) and (member not in modules_to_skip):
            yield from _find_doctests_in_obj(finder,
                                             member, _is_public_name)


def _find_doctests_in_obj(finder, obj, criteria=None):
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
            if _file_from_cugraph(member) and _is_python_module(member):
                _find_doctests_in_obj(finder, member, criteria)
        if inspect.isfunction(member):
            yield from _find_doctests_in_docstring(finder, member)
        if inspect.isclass(member):
            if _module_from_cugraph(member):
                yield from _find_doctests_in_docstring(finder, member)


def _find_doctests_in_docstring(finder, member):
    for docstring in finder.find(member):
        if docstring.examples:
            yield docstring


def _fetch_doctests():
    finder = doctest.DocTestFinder()
    yield from _find_modules_in_obj(finder, cugraph, _is_public_name)


class TestDoctests:
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
        optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        runner = doctest.DocTestRunner(optionflags=optionflags)
        np.random.seed(6)
        globs = dict(cudf=cudf, np=np, cugraph=cugraph, datasets_path=datasets,
                     scipy=scipy, pd=pd)
        docstring.globs = globs

        # FIXME: A 11.4 bug causes ktruss to crash in that
        # environment. Skip docstring test if the cuda version is either
        # 11.2 or 11.4. See ktruss_subgraph.py
        if docstring.name == 'ktruss_subgraph':
            if cuda.runtime.get_version() == (11, 4):
                return

        # Capture stdout and include failing outputs in the traceback.
        doctest_stdout = io.StringIO()
        with contextlib.redirect_stdout(doctest_stdout):
            runner.run(docstring)
            results = runner.summarize()
        assert not results.failed, (
            f"{results.failed} of {results.attempted} doctests failed for "
            f"{docstring.name}:\n{doctest_stdout.getvalue()}"
        )
