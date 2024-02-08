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
from pathlib import Path

from nx_cugraph.scripts.print_tree import create_tree, tree_lines


def replace_body(text, match, new_body):
    start, stop = match.span("body")
    return text[:start] + new_body + text[stop:]


def main(file):
    """``file`` must be readable and writable, so use mode ``"a+"``"""
    file.seek(0)
    text = file.read()
    tree = create_tree()
    # Algorithms
    match = re.search(
        r"### Algorithms\n(?P<preamble>.*?)<pre>\n(?P<body>.*?)\n</pre>",
        text,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError("Algorithms section not found!")
    lines = []
    for key, val in tree["algorithms"].items():
        lines.append(key)
        lines.extend(tree_lines(val, parents=("algorithms", key)))
    text = replace_body(text, match, "\n".join(lines))
    # Generators
    match = re.search(
        r"### Generators\n(?P<preamble>.*?)<pre>\n(?P<body>.*?)\n</pre>",
        text,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError("Generators section not found!")
    lines = []
    for key, val in tree["generators"].items():
        lines.append(key)
        lines.extend(tree_lines(val, parents=("generators", key)))
    text = replace_body(text, match, "\n".join(lines))
    # Other
    match = re.search(
        r"### Other\n(?P<preamble>.*?)<pre>\n(?P<body>.*?)\n</pre>",
        text,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError("Other section not found!")
    lines = []
    for key, val in tree.items():
        if key in {"algorithms", "generators"}:
            continue
        lines.append(key)
        lines.extend(tree_lines(val, parents=(key,)))
    text = replace_body(text, match, "\n".join(lines))
    # Now overwrite README.md
    file.truncate(0)
    file.write(text)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Update README.md to show NetworkX functions implemented by nx-cugraph"
    )
    parser.add_argument("readme_filename", help="Path to the README.md file")
    args = parser.parse_args()
    with Path(args.readme_filename).open("a+") as f:
        main(f)
