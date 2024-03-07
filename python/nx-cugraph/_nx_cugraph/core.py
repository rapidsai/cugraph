# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
"""Utilities to help keep _nx_cugraph up to date."""


def get_functions():
    from nx_cugraph.interface import BackendInterface
    from nx_cugraph.utils import networkx_algorithm

    return {
        key: val
        for key, val in vars(BackendInterface).items()
        if isinstance(val, networkx_algorithm)
    }


def get_additional_docs(functions=None):
    if functions is None:
        functions = get_functions()
    return {key: val.extra_doc for key, val in functions.items() if val.extra_doc}


def get_additional_parameters(functions=None):
    if functions is None:
        functions = get_functions()
    return {key: val.extra_params for key, val in functions.items() if val.extra_params}


def update_text(text, lines_to_add, target, indent=" " * 8):
    begin = f"# BEGIN: {target}\n"
    end = f"# END: {target}\n"
    start = text.index(begin)
    stop = text.index(end)
    to_add = "\n".join([f"{indent}{line}" for line in lines_to_add])
    return f"{text[:start]}{begin}{to_add}\n{indent}{text[stop:]}"


def dict_to_lines(d, *, indent=""):
    for key in sorted(d):
        val = d[key]
        if "\n" not in val:
            yield f"{indent}{key!r}: {val!r},"
        else:
            yield f"{indent}{key!r}: ("
            *lines, last_line = val.split("\n")
            for line in lines:
                line += "\n"
                yield f"    {indent}{line!r}"
            yield f"    {indent}{last_line!r}"
            yield f"{indent}),"


def main(filepath):
    from pathlib import Path

    filepath = Path(filepath)
    with filepath.open() as f:
        orig_text = f.read()
    text = orig_text

    # Update functions
    functions = get_functions()
    to_add = [f'"{name}",' for name in sorted(functions)]
    text = update_text(text, to_add, "functions")

    # Update additional_docs
    additional_docs = get_additional_docs(functions)
    to_add = list(dict_to_lines(additional_docs))
    text = update_text(text, to_add, "additional_docs")

    # Update additional_parameters
    additional_parameters = get_additional_parameters(functions)
    to_add = []
    for name in sorted(additional_parameters):
        params = additional_parameters[name]
        to_add.append(f"{name!r}: {{")
        to_add.extend(dict_to_lines(params, indent=" " * 4))
        to_add.append("},")
    text = update_text(text, to_add, "additional_parameters")
    return text
