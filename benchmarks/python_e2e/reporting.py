# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

from os import path
import csv

import numpy as np


def __namify_dict(d):
    """
    Return a string repr for a dictionary suitable for using in a benchmark
    name.
    """
    strings = []

    for (key, val) in d.items():
        if type(val) in [float, np.float64, np.float32]:
            val = f"{val:.6}"
        strings.append(f"{key}:{val}")

    return ",".join(strings)


def generate_console_report(benchmark_result_list):
    """
    Return a string suitable for printing to the console containing the
    benchmark run results.
    """
    retstring = ""

    # Assume results are ordered based on the order they were run, which is
    # the graph_create run, then a run of each algo.
    r = benchmark_result_list[0]
    name = f"{r.name}({__namify_dict(r.params)})"
    space = " " * (70 - len(name))
    retstring += f"{name}{space}{r.runtime:.6}\n"

    remaining_results = benchmark_result_list[1:]

    for r in remaining_results:
        retstring += f"{'-'*80}\n"
        name = f"{r.name}({__namify_dict(r.params)})"
        space = " " * (70 - len(name))
        retstring += f"{name}{space}{r.runtime:.6}\n"

    return retstring

def update_csv_report(csv_results_file, benchmark_result_list, ngpus):
    """
    Update (or create if DNE) csv_results_file as a CSV file containing the
    benchmark results.  Rows are the runtimes, columns are number of GPUs.
    """
    new_times = {}
    new_names = set()
    names_from_csv = set()

    ngpus_key = f"ngpus_{ngpus}"
    all_fields = set(["name", ngpus_key])

    for r in benchmark_result_list:
        name = r.name
        new_names.add(name)
        new_times[name] = f"{r.runtime:.6}"

    rows = []

    # Read in rows from an existing CSV, then add the new time for the current
    # run with a (possibly new) number of GPUs.
    # This also accumulates all the unique field names.
    if path.exists(csv_results_file):
        with open(csv_results_file) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                algo_name_from_csv = row["name"]
                names_from_csv.add(algo_name_from_csv)
                if algo_name_from_csv in new_times:
                    row[ngpus_key] = new_times[algo_name_from_csv]
                # Ensure all fields in each row are added
                [all_fields.add(f) for f in row.keys()]
                rows.append(row)

    # Add any new algos that were not present in the existing CSV. If there was
    # no existing CSV, then this is all the algos ran in this run.
    for name in new_names - names_from_csv:
        rows.append({"name": name,
                     ngpus_key: new_times[name],
                    })

    with open(csv_results_file, "w") as csv_file:
        field_names = sorted(all_fields)
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
