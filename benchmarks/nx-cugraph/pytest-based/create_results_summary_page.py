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


import re
import pathlib
import json
import platform
import psutil
import socket
import subprocess


def compute_perf_vals(cugraph_runtime, networkx_runtime):
    speedup_string = f"{networkx_runtime / cugraph_runtime:.3f}X"
    delta = networkx_runtime - cugraph_runtime
    if abs(delta) < 1:
        if abs(delta) < 0.001:
            units = "us"
            delta *= 1e6
        else:
            units = "ms"
            delta *= 1e3
    else:
        units = "s"
    delta_string = f"{delta:.3f}{units}"

    return (speedup_string, delta_string)


def get_system_info():
    print(f"<p>Hostname        : {socket.gethostname()}</p>")
    print(
        f'<p class="text-indent"">Operating System: {platform.system()} {platform.release()}</p>'
    )
    print(f'<p class="text-indent">Kernel Version  : {platform.version()}</p>')
    with open("/proc/cpuinfo") as f:
        print(
            f'<p>CPU        : {next(line.strip().split(": ")[1] for line in f if "model name" in line)} ({psutil.cpu_count(logical=False)} cores)</p>'
        )
    print(
        f'<p class="text-indent">Memory       : {round(psutil.virtual_memory().total / (1024 ** 3), 2)} GB</p>'
    )
    try:
        gpu_info = (
            subprocess.check_output(
                "nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader",
                shell=True,
            )
            .decode()
            .strip()
        )
        if gpu_info:
            gpus = gpu_info.split("\n")
            num_gpus = len(gpus)
            first_gpu = gpus[0]  # Get the information for the first GPU
            gpu_name, mem_total, _, _ = first_gpu.split(",")
            print(
                f"<p>GPU          : {num_gpus} x {gpu_name.strip()} ({round(int(mem_total.strip().split()[0]) / (1024), 2)} GB)</p>"
            )
        else:
            print("<p>No GPU found or unable to query GPU details.</p>")
    except subprocess.CalledProcessError:
        print("<p>Failed to execute nvidia-smi. No GPU information available.</p>")


if __name__ == "__main__":
    logs_dir = pathlib.Path("logs")

    dataset_patt = re.compile(".*ds=([\w-]+).*")
    backend_patt = re.compile(".*backend=(\w+).*")
    k_patt = re.compile(".*k=(10*).*")

    # Organize all benchmark runs by the following hierarchy: algo -> backend -> dataset
    benchmarks = {}

    # Populate benchmarks dir from .json files
    for json_file in logs_dir.glob("*.json"):
        # print(f"READING {json_file}")
        try:
            data = json.loads(open(json_file).read())
        except json.decoder.JSONDecodeError:
            # print(f"PROBLEM READING {json_file}, skipping.")
            continue

        for benchmark_run in data["benchmarks"]:
            # example name: "bench_triangles[ds=netscience-backend=cugraph-preconverted]"
            name = benchmark_run["name"]

            algo_name = name.split("[")[0]
            if algo_name.startswith("bench_"):
                algo_name = algo_name[6:]
            # special case for betweenness_centrality
            match = k_patt.match(name)
            if match is not None:
                algo_name += f", k={match.group(1)}"

            match = dataset_patt.match(name)
            if match is None:
                raise RuntimeError(
                    f"benchmark name {name} in file {json_file} has an unexpected format"
                )
            dataset = match.group(1)
            if dataset.endswith("-backend"):
                dataset = dataset[:-8]

            match = backend_patt.match(name)
            if match is None:
                raise RuntimeError(
                    f"benchmark name {name} in file {json_file} has an unexpected format"
                )
            backend = match.group(1)
            if backend == "None":
                backend = "networkx"

            runtime = benchmark_run["stats"]["mean"]
            benchmarks.setdefault(algo_name, {}).setdefault(backend, {})[
                dataset
            ] = runtime

    # dump HTML table
    ordered_datasets = [
        "netscience",
        "email_Eu_core",
        "cit-patents",
        "hollywood",
        "soc-livejournal1",
    ]

    print(
        """
    <html>
    <head>
    <style>
        table {
            table-layout: fixed;
            width: 100%;
            border-collapse: collapse;
        }
        tbody tr:nth-child(odd) {
            background-color: #ffffff;
        }
        tbody tr:nth-child(even) {
            background-color: #d3d3d3;
        }
        tbody td {
            text-align: center;
            color: black;
        }
        th,
        td {
            padding: 10px;
        }
        .footer {
            background-color: #f1f1f1;
            padding: 10px;
            font-size: 12px;
            color: black;
            width: 100%;
        }
        .text-indent {
            text-indent: 20px; /* Indents the first line of the text by 30px */
        }
    </style>
    </head>
    <table>
    <thead>
    <tr>
        <th></th>"""
    )
    for ds in ordered_datasets:
        print(f"      <th>{ds}</th>")
    print(
        """   </tr>
    </thead>
    <tbody>
    """
    )

    for algo_name in benchmarks:
        algo_runs = benchmarks[algo_name]
        print("   <tr>")
        print(f"      <td>{algo_name}</td>")
        # Proceed only if any results are present for both cugraph and NX
        if "cugraph" in algo_runs and "networkx" in algo_runs:
            cugraph_algo_runs = algo_runs["cugraph"]
            networkx_algo_runs = algo_runs["networkx"]
            datasets_in_both = set(cugraph_algo_runs).intersection(networkx_algo_runs)

            # populate the table with speedup results for each dataset in the order
            # specified in ordered_datasets. If results for a run using a dataset
            # are not present for both cugraph and NX, output an empty cell.
            for dataset in ordered_datasets:
                if dataset in datasets_in_both:
                    cugraph_runtime = cugraph_algo_runs[dataset]
                    networkx_runtime = networkx_algo_runs[dataset]
                    (speedup, runtime_delta) = compute_perf_vals(
                        cugraph_runtime=cugraph_runtime,
                        networkx_runtime=networkx_runtime,
                    )
                    print(f"      <td>{speedup}<br>{runtime_delta}</td>")
                else:
                    print(f"      <td></td>")

        # If a comparison between cugraph and NX cannot be made, output empty cells
        # for each dataset
        else:
            for _ in range(len(ordered_datasets)):
                print("      <td></td>")

        print("   </tr>")

    print(
        """
    </tbody>\n</table>
    <div class="footer">"""
    )
    get_system_info()
    print("""</div>\n</html>""")
