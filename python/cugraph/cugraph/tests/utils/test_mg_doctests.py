# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import cugraph
import cudf
from cugraph.testing import utils

datasets = utils.RAPIDS_DATASET_ROOT_DIR_PATH


def _is_public_name(name):
    return not name.startswith("_")


def _is_python_module(member):
    member_file = getattr(member, "__file__", "")
    return os.path.splitext(member_file)[1] == ".py"


def _module_from_library(member, libname):
    return libname in getattr(member, "__file__", "")


def _find_doctests_in_docstring(finder, member):
    for docstring in finder.find(member):
        if docstring.examples:
            yield docstring


def _find_doctests_in_obj(finder, obj, obj_name, criteria=None):
    """Find all doctests in a module or class.
    Parameters
    ----------
    finder : doctest.DocTestFinder
        The DocTestFinder object to use.

    obj : module or class
        The object to search for docstring examples.

    obj_name : string
        Used for ensuring a module is part of the object.
        To be passed into _module_from_library.

    criteria : callable, optional

    Yields
    ------
    doctest.DocTest
        The next doctest found in the object.
    """
    for name, member in inspect.getmembers(obj, inspect.isfunction):
        if criteria is not None and not criteria(name):
            continue
        if inspect.ismodule(member):
            yield from _find_doctests_in_obj(finder, member, obj_name, criteria)
        if inspect.isfunction(member):
            yield from _find_doctests_in_docstring(finder, member)
        if inspect.isclass(member):
            if _module_from_library(member, obj_name):
                yield from _find_doctests_in_docstring(finder, member)


def _fetch_doctests():
    finder = doctest.DocTestFinder()
    yield from _find_doctests_in_obj(finder, cugraph.dask, "dask", _is_public_name)


@pytest.fixture(
    scope="module", params=_fetch_doctests(), ids=lambda docstring: docstring.name
)
def docstring(request):
    return request.param


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

    def test_docstring(self, dask_client, docstring):
        # We ignore differences in whitespace in the doctest output, and enable
        # the use of an ellipsis "..." to match any string in the doctest
        # output. An ellipsis is useful for, e.g., memory addresses or
        # imprecise floating point values.
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
        assert not results.failed, (
            f"{results.failed} of {results.attempted} doctests failed for "
            f"{docstring.name}:\n{doctest_stdout.getvalue()}"
        )
