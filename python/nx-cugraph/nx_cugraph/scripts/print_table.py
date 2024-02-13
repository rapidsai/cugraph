#!/usr/bin/env python
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
import argparse
import sys
from collections import namedtuple

from networkx.utils.backends import _registered_algorithms as algos

from _nx_cugraph import get_info
from nx_cugraph.interface import BackendInterface


def get_funcpath(func):
    return f"{func.__module__}.{func.__name__}"


def get_path_to_name():
    return {
        get_funcpath(algos[funcname]): funcname
        for funcname in get_info()["functions"].keys() & algos.keys()
    }


Info = namedtuple(
    "Info",
    "networkx_path, dispatch_name, version_added, plc, is_incomplete, is_different",
)


def get_path_to_info(path_to_name=None, version_added_sep=".", plc_sep="/"):
    if path_to_name is None:
        path_to_name = get_path_to_name()
    rv = {}
    for funcpath in sorted(path_to_name):
        funcname = path_to_name[funcpath]
        cufunc = getattr(BackendInterface, funcname)
        plc = plc_sep.join(sorted(cufunc._plc_names)) if cufunc._plc_names else ""
        version_added = cufunc.version_added.replace(".", version_added_sep)
        is_incomplete = cufunc.is_incomplete
        is_different = cufunc.is_different
        rv[funcpath] = Info(
            funcpath, funcname, version_added, plc, is_incomplete, is_different
        )
    return rv


def main(path_to_info=None, *, file=sys.stdout):
    if path_to_info is None:
        path_to_info = get_path_to_info(version_added_sep=".")
    lines = ["networkx_path,dispatch_name,version_added,plc,is_incomplete,is_different"]
    lines.extend(",".join(map(str, info)) for info in path_to_info.values())
    text = "\n".join(lines)
    print(text, file=file)
    return text


def get_argumentparser(add_help=True):
    return argparse.ArgumentParser(
        description="Print info about functions implemented by nx-cugraph as CSV",
        add_help=add_help,
    )


if __name__ == "__main__":
    parser = get_argumentparser()
    args = parser.parse_args()
    main()
