import doctest
import inspect
import os

import numpy as np
import pandas as pd
import scipy
import pytest

import cugraph
from cugraph.tests import utils
import cudf
import pathlib


modules_to_skip = ["dask", "proto", "raft"]
datasets = pathlib.Path(cugraph.__path__[0]).parent.parent.parent / "datasets"

def _is_public_name(parent, name, member):
    return not name.startswith("_")

def _is_python_module(parent, name, member):
    return os.path.splitext(member.__file__)[1] == '.py'

def _module_from_cugraph(parent, name, member):
    return 'cugraph' in member.__module__

def _file_from_cugraph(parent, name, member):
    return 'cugraph' in member.__file__ 



def _find_modules_in_obj(finder, obj, criteria=None):
    for name, member in inspect.getmembers(obj):
        if criteria is not None and not criteria(obj, name, member):
            continue
        if inspect.ismodule(member):    
            yield from _find_members_in_module(finder, member, criteria=_is_public_name)

def _find_members_in_module(finder, obj, criteria=None):
    for name, member in inspect.getmembers(obj):
        if criteria is not None and not criteria(obj, name, member):
            continue

        if inspect.ismodule(member) and (member not in modules_to_skip):
            if _file_from_cugraph(obj, name, member) and _is_python_module(obj, name, member):
                _find_members_in_module(finder, member, criteria)
        if inspect.isfunction(member):
            yield from _find_examples_in_docstring(finder, member)
        if inspect.isclass(member):
            if _module_from_cugraph(obj, name, member):
                yield from _find_examples_in_docstring(finder, member)

def _find_examples_in_docstring(finder, member):
    for docstring in finder.find(member):
        if docstring.examples:
            yield docstring

def _fetch_doctests():
    finder = doctest.DocTestFinder()
    yield from _find_members_in_module(finder, cugraph.link_analysis, criteria=_is_public_name)
    #yield from _find_modules_in_obj(finder, cugraph, criteria=_is_public_name)


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
        globs = dict(cudf=cudf, np=np, cugraph=cugraph, datasets=datasets, scipy=scipy, pd=pd)
        docstring.globs = globs
        #print(docstring)
        runner.run(docstring)
        results = runner.summarize()
        if results.failed:
            raise AssertionError(results)
