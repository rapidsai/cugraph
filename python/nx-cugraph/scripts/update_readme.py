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
import urllib.request
import zlib
from collections import namedtuple
from pathlib import Path
from warnings import warn

_objs_file_url = "https://networkx.org/documentation/stable/objects.inv"

# See: https://sphobjinv.readthedocs.io/en/stable/syntax.html
DocObject = namedtuple(
    "DocObject",
    "name, domain, role, priority, uri, displayname",
)


def parse_docobject(line):
    left, right = line.split(":")
    name, domain = left.rsplit(" ", 1)
    role, priority, uri, displayname = right.split(" ", 3)
    if displayname == "-":
        displayname = name
    if uri.endswith("$"):
        uri = uri[:-1] + name
    return DocObject(name, domain, role, priority, uri, displayname)


def replace_body(text, match, new_body):
    start, stop = match.span("body")
    return text[:start] + new_body + text[stop:]


# NetworkX isn't perfectly intersphinx-compatible, so manually specify some urls.
# See: https://github.com/networkx/networkx/issues/7278
MANUAL_OBJECT_URLS = {
    "networkx.algorithms.centrality.betweenness": (
        "https://networkx.org/documentation/stable/reference/"
        "algorithms/centrality.html#shortest-path-betweenness"
    ),
    "networkx.algorithms.centrality.degree_alg": (
        "https://networkx.org/documentation/stable/reference/"
        "algorithms/centrality.html#degree"
    ),
    "networkx.algorithms.centrality.eigenvector": (
        "https://networkx.org/documentation/stable/reference/"
        "algorithms/centrality.html#eigenvector"
    ),
    "networkx.algorithms.centrality.katz": (
        "https://networkx.org/documentation/stable/reference/"
        "algorithms/centrality.html#eigenvector"
    ),
    "networkx.algorithms.components.connected": (
        "https://networkx.org/documentation/stable/reference/"
        "algorithms/component.html#connectivity"
    ),
    "networkx.algorithms.components.weakly_connected": (
        "https://networkx.org/documentation/stable/reference/"
        "algorithms/component.html#weak-connectivity"
    ),
    "networkx.classes": (
        "https://networkx.org/documentation/stable/reference/classes/index.html"
    ),
}


def main(readme_file, objects_filename):
    """``readme_file`` must be readable and writable, so use mode ``"a+"``"""
    from nx_cugraph.scripts.print_tree import create_tree, tree_lines

    # Use the `objects.inv` file to determine URLs. For details about this file, see:
    # https://sphobjinv.readthedocs.io/en/stable/syntax.html
    # We might be better off using a library like that, but roll our own for now.
    with Path(objects_filename).open("rb") as objects_file:
        line = objects_file.readline()
        if line != b"# Sphinx inventory version 2\n":
            raise RuntimeError(f"Bad line in objects.inv:\n\n{line}")
        line = objects_file.readline()
        if line != b"# Project: NetworkX\n":
            raise RuntimeError(f"Bad line in objects.inv:\n\n{line}")
        line = objects_file.readline()
        if not line.startswith(b"# Version: "):
            raise RuntimeError(f"Bad line in objects.inv:\n\n{line}")
        line = objects_file.readline()
        if line != b"# The remainder of this file is compressed using zlib.\n":
            raise RuntimeError(f"Bad line in objects.inv:\n\n{line}")
        zlib_data = objects_file.read()
    objects_text = zlib.decompress(zlib_data).decode().strip()
    objects_list = [parse_docobject(line) for line in objects_text.split("\n")]
    doc_urls = {
        obj.name: "https://networkx.org/documentation/stable/" + obj.uri
        for obj in objects_list
    }
    if len(objects_list) != len(doc_urls):
        raise RuntimeError("Oops; duplicate names found in objects.inv")

    def get_payload(info, **kwargs):
        path = "networkx." + info.networkx_path
        subpath, name = path.rsplit(".", 1)
        # Many objects are referred to in modules above where they are defined.
        while subpath:
            path = f"{subpath}.{name}"
            if path in doc_urls:
                return f'<a href="{doc_urls[path]}">{name}</a>'
            subpath = subpath.rsplit(".", 1)[0]
        warn(f"Unable to find URL for {name!r}: {path}", stacklevel=0)
        return name

    def get_payload_internal(keys):
        path = "networkx." + ".".join(keys)
        name = keys[-1]
        if path in doc_urls:
            return f'<a href="{doc_urls[path]}">{name}</a>'
        path2 = "reference/" + "/".join(keys)
        if path2 in doc_urls:
            return f'<a href="{doc_urls[path2]}">{name}</a>'
        if path in MANUAL_OBJECT_URLS:
            return f'<a href="{MANUAL_OBJECT_URLS[path]}">{name}</a>'
        warn(f"Unable to find URL for {name!r}: {path}", stacklevel=0)
        return name

    readme_file.seek(0)
    text = readme_file.read()
    tree = create_tree(get_payload=get_payload)
    # Algorithms
    match = re.search(
        r"### .Algorithms(?P<preamble>.*?)<pre>\n(?P<body>.*?)\n</pre>",
        text,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError("Algorithms section not found!")
    lines = []
    for key, val in tree["algorithms"].items():
        lines.append(get_payload_internal(("algorithms", key)))
        lines.extend(
            tree_lines(
                val,
                parents=("algorithms", key),
                get_payload_internal=get_payload_internal,
            )
        )
    text = replace_body(text, match, "\n".join(lines))
    # Generators
    match = re.search(
        r"### .Generators(?P<preamble>.*?)<pre>\n(?P<body>.*?)\n</pre>",
        text,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError("Generators section not found!")
    lines = []
    for key, val in tree["generators"].items():
        lines.append(get_payload_internal(("generators", key)))
        lines.extend(
            tree_lines(
                val,
                parents=("generators", key),
                get_payload_internal=get_payload_internal,
            )
        )
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
        lines.append(get_payload_internal((key,)))
        lines.extend(
            tree_lines(val, parents=(key,), get_payload_internal=get_payload_internal)
        )
    text = replace_body(text, match, "\n".join(lines))
    # Now overwrite README.md
    readme_file.truncate(0)
    readme_file.write(text)
    return text


def find_or_download_objs_file(objs_file_dir):
    """Return the path to <objs_file_dir>/objects.inv and download it if necessary.

    Download objects.inv from _objs_file_url if it does not already exist.
    """
    objs_file_path = objs_file_dir / "objects.inv"
    if not objs_file_path.exists():
        request = urllib.request.Request(_objs_file_url)
        with (
            urllib.request.urlopen(request) as response,
            Path(objs_file_path).open("wb") as out,
        ):
            out.write(response.read())
    return objs_file_path


if __name__ == "__main__":
    # This script imports a nx_cugraph script module, which imports nx_cugraph
    # runtime dependencies. The script module does not need the runtime deps,
    # so stub them out to avoid installing them.
    class Stub:
        def __getattr__(self, *args, **kwargs):
            return Stub()

        def __call__(self, *args, **kwargs):
            return Stub()

    import sys

    sys.modules["cupy"] = Stub()
    sys.modules["numpy"] = Stub()
    sys.modules["pylibcugraph"] = Stub()

    parser = argparse.ArgumentParser(
        "Update README.md to show NetworkX functions implemented by nx-cugraph"
    )
    parser.add_argument("readme_filename", help="Path to the README.md file")
    parser.add_argument(
        "networkx_objects",
        nargs="?",
        default=None,
        help="Optional path to the objects.inv file from the NetworkX docs. Default is "
        "the objects.inv file in the directory containing the specified README.md. If "
        "an objects.inv file does not exist in that location, one will be downloaded "
        "and saved to that location.",
    )
    args = parser.parse_args()

    readme_filename = args.readme_filename
    readme_path = Path(readme_filename)
    objects_filename = args.networkx_objects
    if objects_filename is None:
        objects_filename = find_or_download_objs_file(readme_path.parent)

    with readme_path.open("a+") as readme_file:
        main(readme_file, objects_filename)
