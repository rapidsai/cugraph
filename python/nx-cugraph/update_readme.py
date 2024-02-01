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
import io
import re

from nx_cugraph.scripts.print_table import main as nxcg_print_table

import pandas as pd

table_header_string = "| feature/algo | release/target version |"
table_header_patt = re.compile(r"\| feature/algo[\s]+\| release/target version[\s]+\|")
nxcg_algo_col_name = "dispatch_name"
readme_algo_col_name = "feature/algo"
nxcg_version_col_name = "version_added"
readme_version_col_name = "release/target version"


def get_current_nxcg_data():
    """
    Returns a DataFrame containing all meta-data from the current nx_cugraph package.
    """
    buf = io.StringIO()
    nxcg_print_table(file=buf)
    buf.seek(0)
    return pd.read_csv(buf, dtype={nxcg_version_col_name: str})


def get_readme_sections(readme_file_name):
    """
    Returns the README as three lists of strings: (before_table, table, after_table)
    """
    assert readme_file_name.endswith(".md")

    before_table = []
    table = []
    after_table = []

    with open(readme_file_name) as fd:
        lines = iter([ln.rstrip() for ln in fd.readlines()])
        line = next(lines, None)

        # everything before the markdown table
        while line is not None and not table_header_patt.fullmatch(line):
            before_table.append(line)
            line = next(lines, None)

        if line is not None and table_header_patt.fullmatch(line):
            # table body
            while line is not None and line.startswith("|"):
                table.append(line)
                line = next(lines, None)

            # everything after the table
            while line is not None:
                after_table.append(line)
                line = next(lines, None)

        else:
            raise RuntimeError(
                "Could not find start of table matching "
                f"'{table_header_string}' in {readme_file_name}"
            )

    return (before_table, table, after_table)


def get_readme_table_data(table_lines):
    """
    Returns a DataFrame containing all meta-data extracted from the markdown
    table text passed in as a list of strings.
    """
    csv_buf = io.StringIO()
    lines = iter(table_lines)
    line = next(lines, None)

    # process header
    # Separate markdown line containing " | " delims and remove leading
    # and trailing empty fields resulting from start/end "|" borders
    fields = [f.strip() for f in line.split("|") if f]
    print(*fields, sep=",", file=csv_buf)

    # Assume header underline line and consume it
    line = next(lines, None)
    assert line.startswith("|:-") or line.startswith("| -")

    # Read the table body
    line = next(lines, None)
    while line is not None and line.startswith("|"):
        fields = [f.strip() for f in line.split("|") if f]
        print(*fields, sep=",", file=csv_buf)
        line = next(lines, None)

    csv_buf.seek(0)
    return pd.read_csv(csv_buf, dtype={readme_version_col_name: str})


def main(readme_file_name="README.md"):
    nxcg_data = get_current_nxcg_data()
    (before_table_lines, table_lines, after_table_lines) = get_readme_sections(
        readme_file_name
    )
    readme_data = get_readme_table_data(table_lines)

    # Use only the data needed for the README
    nxcg_data_for_readme = nxcg_data[
        [nxcg_algo_col_name, nxcg_version_col_name]
    ].rename(
        {
            nxcg_algo_col_name: readme_algo_col_name,
            nxcg_version_col_name: readme_version_col_name,
        },
        axis=1,
    )

    # Update the readme data with the latest nxcg data. This will add new algos
    # to the readme data and replace any old version values in the readme data
    # with current nxcg version values.
    merged = readme_data.merge(
        nxcg_data_for_readme,
        how="outer",
        on=readme_algo_col_name,
    )
    x = readme_version_col_name + "_x"
    y = readme_version_col_name + "_y"
    merged[readme_version_col_name] = merged[y].fillna(merged[x])
    merged.drop([x, y], axis=1, inplace=True)
    merged.sort_values(by=readme_algo_col_name, inplace=True)

    # Rewrite the README with the updated table
    with open(readme_file_name, "w") as fd:
        print("\n".join(before_table_lines), file=fd)
        print(merged.to_markdown(index=False), file=fd)
        print("\n".join(after_table_lines), file=fd)


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
