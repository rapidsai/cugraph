# Copyright (c) 2021, NVIDIA CORPORATION.
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
    space = " " * (30 - len(name))
    retstring += f"{name}{space}{r.runtime:.6}\n"

    remaining_results = benchmark_result_list[1:]

    for r in remaining_results:
        retstring += f"{'-'*60}\n"
        name = f"{r.name}({__namify_dict(r.params)})"
        space = " " * (30 - len(name))
        retstring += f"{name}{space}{r.runtime:.6}\n"

    return retstring


# FIXME: refactor this out since it is Graph500-specific
def generate_graph500_console_report(benchmark_result_list):
    """
    Return a string suitable for printing to the console containing the
    benchmark run results.
    """
    retstring = ""

    # Assume results are ordered based on the order they were run, which for
    # Graph500 is a single graph_create run, then a run for each search key for
    # BFS, then for each search key for SSSP.
    r = benchmark_result_list[0]
    name = f"{r.name}({__namify_dict(r.params)})"
    space = " " * (30 - len(name))
    retstring += f"{name}{space}{r.runtime:.6}\n"

    remaining_results = benchmark_result_list[1:]
    half = len(remaining_results) // 2
    bfs_results = remaining_results[:half]
    sssp_results = remaining_results[half:]

    for results in [bfs_results, sssp_results]:
        retstring += f"{'-'*60}\n"
        for r in results:
            name = f"{r.name}({__namify_dict(r.params)})"
            space = " " * (30 - len(name))
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


# FIXME: refactor this out since it is Graph500-specific
def update_graph500_csv_report(csv_results_file, benchmark_result_list, ngpus):
    """
    Update (or create if DNE) csv_results_file as a CSV file containing the
    benchmark results.  Rows are the overall avergage for each of the three
    timed kernels, columns are number of GPUs.
    """
    times = {}
    all_times = {}
    all_names = set()

    ngpus_key = f"ngpus_{ngpus}"

    for r in benchmark_result_list:
        name = r.name
        all_names.add(name)
        min_time_name = f"{name}_min"
        max_time_name = f"{name}_max"

        min_time = times.get(min_time_name)

        if r.runtime > times.get(max_time_name, 0):
            times[max_time_name] = r.runtime
        if min_time is None or r.runtime < min_time:
            times[min_time_name] = r.runtime
        all_times.setdefault(name, []).append(r.runtime)

    for name in all_times:
        mean_time_name = f"{name}_mean"
        times[mean_time_name] = sum(all_times[name]) / len(all_times[name])

    rows = []

    if path.exists(csv_results_file):
        with open(csv_results_file) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                row[ngpus_key] = times[row["name"]]
                rows.append(row)

    else:
        for name in all_names:
            min_time_name = f"{name}_min"
            max_time_name = f"{name}_max"
            mean_time_name = f"{name}_mean"

            rows.append({"name": min_time_name,
                         ngpus_key: times[min_time_name],
                         })
            rows.append({"name": max_time_name,
                         ngpus_key: times[max_time_name],
                         })
            rows.append({"name": mean_time_name,
                         ngpus_key: times[mean_time_name],
                         })

    with open(csv_results_file, "w") as csv_file:
        field_names = sorted(rows[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
