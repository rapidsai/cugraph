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

import doctest
import inspect
import os
import pathlib

import numpy as np
import pandas as pd
import scipy
import pytest

import cugraph
import cudf


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
            yield from _find_members_in_module(finder,
                                               member, _is_public_name)


def _find_members_in_module(finder, obj, criteria=None):
    for name, member in inspect.getmembers(obj):
        if criteria is not None and not criteria(name):
            continue

        if inspect.ismodule(member):
            if _file_from_cugraph(member) and _is_python_module(member):
                _find_members_in_module(finder, member, criteria)
        if inspect.isfunction(member):
            yield from _find_examples_in_docstring(finder, member)
        if inspect.isclass(member):
            if _module_from_cugraph(member):
                yield from _find_examples_in_docstring(finder, member)


def _find_examples_in_docstring(finder, member):
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
        optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        runner = doctest.DocTestRunner(optionflags=optionflags)
        globs = dict(cudf=cudf, np=np, cugraph=cugraph, datasets_path=datasets,
                     scipy=scipy, pd=pd)
        docstring.globs = globs
        runner.run(docstring)
        results = runner.summarize()
        if results.failed:
            raise AssertionError(results)
