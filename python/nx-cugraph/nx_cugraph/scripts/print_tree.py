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
import re
import sys

import networkx as nx

from nx_cugraph.scripts.print_table import get_path_to_info


def add_branch(G, path, extra="", *, skip=0):
    branch = path.split(".")
    prev = ".".join(branch[: skip + 1])
    for i in range(skip + 2, len(branch)):
        cur = ".".join(branch[:i])
        G.add_edge(prev, cur)
        prev = cur
    if extra:
        if not isinstance(extra, str):
            extra = ", ".join(extra)
        path += f" ({extra})"
    G.add_edge(prev, path)


def get_extra(
    info,
    *,
    networkx_path=False,
    dispatch_name=False,
    version_added=False,
    plc=False,
    dispatch_name_if_different=False,
    incomplete=False,
    different=False,
):
    extra = []
    if networkx_path:
        extra.append(info.networkx_path)
    if dispatch_name and (
        not dispatch_name_if_different
        or info.dispatch_name != info.networkx_path.rsplit(".", 1)[-1]
    ):
        extra.append(info.dispatch_name)
    if version_added:
        v = info.version_added
        if len(v) != 5:
            raise ValueError(f"Is there something wrong with version: {v!r}?")
        extra.append(v[:2] + "." + v[-2:])
    if plc and info.plc:
        extra.append(info.plc)
    if incomplete and info.is_incomplete:
        extra.append("is-incomplete")
    if different and info.is_different:
        extra.append("is-different")
    return extra


def create_tree(
    path_to_info=None,
    *,
    by="networkx_path",
    skip=0,
    networkx_path=False,
    dispatch_name=False,
    version_added=False,
    plc=False,
    dispatch_name_if_different=False,
    incomplete=False,
    different=False,
    prefix="",
):
    if path_to_info is None:
        path_to_info = get_path_to_info()
    if isinstance(by, str):
        by = [by]
    G = nx.DiGraph()
    for info in sorted(
        path_to_info.values(),
        key=lambda x: (*(getattr(x, b) for b in by), x.networkx_path),
    ):
        if not all(getattr(info, b) for b in by):
            continue
        path = prefix + ".".join(getattr(info, b) for b in by)
        extra = get_extra(
            info,
            networkx_path=networkx_path,
            dispatch_name=dispatch_name,
            version_added=version_added,
            plc=plc,
            dispatch_name_if_different=dispatch_name_if_different,
            incomplete=incomplete,
            different=different,
        )
        add_branch(G, path, extra=extra, skip=skip)
    return G


def main(
    path_to_info=None,
    *,
    by="networkx_path",
    networkx_path=False,
    dispatch_name=False,
    version_added=False,
    plc=False,
    dispatch_name_if_different=True,
    incomplete=False,
    different=False,
    file=sys.stdout,
):
    if path_to_info is None:
        path_to_info = get_path_to_info(version_added_sep="-")
    kwargs = {
        "networkx_path": networkx_path,
        "dispatch_name": dispatch_name,
        "version_added": version_added,
        "plc": plc,
        "dispatch_name_if_different": dispatch_name_if_different,
        "incomplete": incomplete,
        "different": different,
    }
    if by == "networkx_path":
        G = create_tree(path_to_info, by="networkx_path", **kwargs)
        text = re.sub(
            r" [A-Za-z_\./]+\.", " ", ("\n".join(nx.generate_network_text(G)))
        )
    elif by == "plc":
        G = create_tree(
            path_to_info, by=["plc", "networkx_path"], prefix="plc-", **kwargs
        )
        text = re.sub(
            "plc-",
            "plc.",
            re.sub(
                r" plc-[A-Za-z_\./]*\.",
                " ",
                "\n".join(nx.generate_network_text(G)),
            ),
        )
    elif by == "version_added":
        G = create_tree(
            path_to_info,
            by=["version_added", "networkx_path"],
            prefix="version_added-",
            **kwargs,
        )
        text = re.sub(
            "version_added-",
            "version: ",
            re.sub(
                r" version_added-[-0-9A-Za-z_\./]*\.",
                " ",
                "\n".join(nx.generate_network_text(G)),
            ),
        ).replace("-", ".")
    else:
        raise ValueError(
            "`by` argument should be one of {'networkx_path', 'plc', 'version_added' "
            f"got: {by}"
        )
    print(text, file=file)
    return text


def get_argumentparser(add_help=True):
    parser = argparse.ArgumentParser(
        "Print a tree showing NetworkX functions implemented by nx-cugraph",
        add_help=add_help,
    )
    parser.add_argument(
        "--by",
        choices=["networkx_path", "plc", "version_added"],
        default="networkx_path",
        help="How to group functions",
    )
    parser.add_argument(
        "--dispatch-name",
        "--dispatch_name",
        action="store_true",
        help="Show the dispatch name in parentheses if different from NetworkX name",
    )
    parser.add_argument(
        "--dispatch-name-always",
        "--dispatch_name_always",
        action="store_true",
        help="Always show the dispatch name in parentheses",
    )
    parser.add_argument(
        "--plc",
        "--pylibcugraph",
        action="store_true",
        help="Show the used pylibcugraph function in parentheses",
    )
    parser.add_argument(
        "--version-added",
        "--version_added",
        action="store_true",
        help="Show the version added in parentheses",
    )
    parser.add_argument(
        "--networkx-path",
        "--networkx_path",
        action="store_true",
        help="Show the full networkx path in parentheses",
    )
    parser.add_argument(
        "--incomplete",
        action="store_true",
        help="Show which functions are incomplete",
    )
    parser.add_argument(
        "--different",
        action="store_true",
        help="Show which functions are different",
    )
    return parser


if __name__ == "__main__":
    parser = get_argumentparser()
    args = parser.parse_args()
    main(
        by=args.by,
        networkx_path=args.networkx_path,
        dispatch_name=args.dispatch_name or args.dispatch_name_always,
        version_added=args.version_added,
        plc=args.plc,
        dispatch_name_if_different=not args.dispatch_name_always,
        incomplete=args.incomplete,
        different=args.different,
    )
