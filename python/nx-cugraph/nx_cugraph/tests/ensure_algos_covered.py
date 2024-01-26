# Copyright (c) 2024, NVIDIA CORPORATION.
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
"""Ensure that all functions wrapped by @networkx_algorithm were called.

This file is run by CI and should not normally be run manually.
"""
import inspect
import json
from pathlib import Path

from nx_cugraph.interface import BackendInterface
from nx_cugraph.utils import networkx_algorithm

with Path("coverage.json").open() as f:
    coverage = json.load(f)

filenames_to_executed_lines = {
    "nx_cugraph/"
    + filename.rsplit("nx_cugraph/", 1)[-1]: set(coverage_info["executed_lines"])
    for filename, coverage_info in coverage["files"].items()
}


def unwrap(func):
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func


def get_func_filename(func):
    return "nx_cugraph" + inspect.getfile(unwrap(func)).rsplit("nx_cugraph", 1)[-1]


def get_func_linenos(func):
    lines, lineno = inspect.getsourcelines(unwrap(func))
    for i, line in enumerate(lines, lineno):
        if ":\n" in line:
            return set(range(i + 1, lineno + len(lines)))
    raise RuntimeError(f"Could not determine line numbers for function {func}")


def has_any_coverage(func):
    return bool(
        filenames_to_executed_lines[get_func_filename(func)] & get_func_linenos(func)
    )


def main():
    no_coverage = set()
    for attr, func in vars(BackendInterface).items():
        if not isinstance(func, networkx_algorithm):
            continue
        if not has_any_coverage(func):
            no_coverage.add(attr)
    if no_coverage:
        msg = "The following algorithms have no coverage: " + ", ".join(
            sorted(no_coverage)
        )
        # Create a border of "!"
        msg = (
            "\n\n"
            + "!" * (len(msg) + 6)
            + "\n!! "
            + msg
            + " !!\n"
            + "!" * (len(msg) + 6)
            + "\n"
        )
        raise AssertionError(msg)
    print("\nSuccess: coverage determined all algorithms were called!\n")


if __name__ == "__main__":
    main()
