import doctest
import inspect
import os

import numpy as np
#import pytest  GOAL is to find all examples w/o pytest

import cugraph
import cudf



modules_to_skip = ["dask", "proto", "raft"]


def is_public_name(parent, name, member):
    return not name.startswith("_")

def is_python_module(parent, name, member):
    return os.path.splitext(member.__file__)[1] == '.py'

def module_from_cugraph(parent, name, member):
    return 'cugraph' in member.__module__

def file_from_cugraph(parent, name, member):
    return 'cugraph' in member.__file__ 



def find_modules_in_obj(finder, obj, criteria=None):
    for name, member in inspect.getmembers(obj):
        if criteria is not None and not criteria(obj, name, member):
            continue
        if inspect.ismodule(member):    
            yield from find_members_in_module(finder, member, criteria=is_public_name)

def find_members_in_module(finder, obj, criteria=None):
    for name, member in inspect.getmembers(obj):
        if criteria is not None and not criteria(obj, name, member):
            continue

        if inspect.ismodule(member) and (member not in modules_to_skip):
            if file_from_cugraph(obj, name, member) and is_python_module(obj, name, member):
                find_members_in_module(finder, member, criteria)
        if inspect.isfunction(member):
            yield from find_examples_in_docstring(finder, member)
        if inspect.isclass(member):
            if module_from_cugraph(obj, name, member):
                yield from find_examples_in_docstring(finder, member)

def find_examples_in_docstring(finder, member):
    for docstring in finder.find(member):
        if docstring.examples:
            yield docstring

def fetch_doctests():
    finder = doctest.DocTestFinder()
    yield from find_modules_in_obj(finder, cugraph, criteria=is_public_name)

#iter = 497
iter = 500
num_tests_found = 0
fetcher = fetch_doctests()
while iter > 0:
    try:
        fetched = next(fetcher)
    except StopIteration:
        break
    if isinstance(fetched, doctest.DocTest):
        print(fetched)
        num_tests_found += len(fetched.examples)
    iter -= 1
print("Total # of tests found: " + str(num_tests_found))