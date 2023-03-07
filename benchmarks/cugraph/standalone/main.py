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

import sys

from cugraph.dask.common.mg_utils import get_visible_devices

from reporting import (generate_console_report,
                       update_csv_report,
                       )

import cugraph_funcs
import cugraph_dask_funcs
from benchmark import BenchmarkRun

import json
from pathlib import Path


def store_results_json(benchmark_dir=None,
                       algo_name=None,
                       algo_time=None,
                       n_gpus=None,
                       scale=None):
    """
    Store all benchmark results in json files
    """
    benchmark_result = {}
    benchmark_result['funcName'] = algo_name
    benchmark_result['result'] = algo_time
    benchmark_result['argNameValuePairs'] = [('scale', scale), ('ngpus', n_gpus)]

    json_object = json.dumps(benchmark_result, indent=4)

    benchmark_dir_path = Path(benchmark_dir)

    with open(f"{benchmark_dir_path}/benchmark_result_scale_{scale}_ngpus_{n_gpus}_{algo_name}.json", "w") as outfile:
        outfile.write(json_object)



def log(s, end="\n"):
    print(s, end=end)
    sys.stdout.flush()


def run(algos,
        scale=None,
        csv_graph_file=None,
        csv_results_file=None,
        unweighted=False,
        symmetric=False,
        edgefactor=None,
        benchmark_dir=None,
        dask_scheduler_file=None,
        rmm_pool_size=None):
    """
    Run the nightly benchmark on cugraph.
    Return True on success, False on failure.
    """
    seed = 42
    n_gpus = len(get_visible_devices())
    if (dask_scheduler_file is None) and (n_gpus < 2):
        funcs = cugraph_funcs
    else:
        funcs = cugraph_dask_funcs
        # For MNMG, find the number of GPUS after the client and
        # and the cluster started
        n_gpus = None

    # Setup the benchmarks to run based on algos specified, or all.
    # Values are either callables, or tuples of (callable, args) pairs.
    benchmarks = {"bfs": funcs.bfs,
                  "sssp": funcs.sssp,
                  "louvain": funcs.louvain,
                  "pagerank": funcs.pagerank,
                  "wcc": funcs.wcc,
                  "katz": funcs.katz,
                  "wcc": funcs.wcc,
                  "hits": funcs.hits,
                  "uniform_neighbor_sample": funcs.uniform_neighbor_sample,
                  "triangle_count": funcs.triangle_count,
                  "eigenvector_centrality": funcs.eigenvector_centrality,
                 }

    if algos:
        invalid_benchmarks = set(algos) - set(benchmarks.keys())
        if invalid_benchmarks:
            raise ValueError("Invalid benchmark(s) specified "
                             f"{invalid_benchmarks}")
        benchmarks_to_run = [benchmarks[b] for b in algos]
    else:
        benchmarks_to_run = list(benchmarks.values())

    # Call the global setup. This is used for setting up Dask, initializing
    # output files/reports, etc.
    log("calling setup...", end="")
    setup_objs = funcs.setup(dask_scheduler_file, rmm_pool_size)

    # If the number of GPUs is None, This is a MNMG run
    # Extract the number of gpus from the client
    if n_gpus is None:
        n_gpus = len(setup_objs[0].scheduler_info()['workers'])
    log("done.")

    try:
        if csv_graph_file:
            log("running read_csv...", end="")
            df = funcs.read_csv(csv_graph_file, 0)
            log("done.")
        elif scale:
            log("running generate_edgelist (RMAT)...", end="")
            df = funcs.generate_edgelist(scale,
                                         edgefactor=edgefactor,
                                         seed=seed,
                                         unweighted=unweighted)
            log("done.")
        else:
            raise ValueError("Must specify either scale or csv_graph_file")

        benchmark = BenchmarkRun(df,
                                 (funcs.construct_graph, (symmetric,)),
                                 benchmarks_to_run,
                                )
        success = benchmark.run()

        algo_name = benchmark.results[1].name
        algo_name = f"benchmarks.{algo_name}"
        algo_time = benchmark.results[1].runtime
        # Generate json files containing the benchmark results
        if benchmark_dir is not None:
            store_results_json(benchmark_dir, algo_name, algo_time, n_gpus, scale)
        
        # Report results
        print(generate_console_report(benchmark.results))
        if csv_results_file:
            update_csv_report(csv_results_file, benchmark.results, n_gpus)

    except:
        success = False
        raise

    finally:
        # Global cleanup
        log("calling teardown...", end="")
        funcs.teardown(*setup_objs)
        log("done.")

    return 0 if success else 1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=None,
                    help="scale factor for the graph edgelist generator "
                    "(num_verts=2**SCALE).")
    ap.add_argument("--csv", type=str, default=None,
                    help="path to CSV file to read instead of generating a "
                    "graph edgelist.")
    ap.add_argument("--unweighted", default=False, action="store_true",
                    help="Generate a graph without weights.")
    ap.add_argument("--algo", action="append",
                    help="Algo to benchmark. May be specified multiple times. "
                    "Default is all algos.")
    ap.add_argument("--dask-scheduler-file", type=str, default=None,
                    help="Dask scheduler file for multi-node configuration.")
    ap.add_argument("--symmetric-graph", default=False, action="store_true",
                    help="Generate a symmetric (undirected) Graph instead of "
                    "a DiGraph.")
    ap.add_argument("--edgefactor", type=int, default=16,
                    help="edge factor for the graph edgelist generator "
                    "(num_edges=num_verts*EDGEFACTOR).")
    ap.add_argument("--benchmark-dir", type=str, default=None,
                    help="directory to store the results in json files")
    ap.add_argument("--rmm-pool-size", type=str, default=None,
                    help="RMM pool size to initialize each worker with")


    args = ap.parse_args()

    exitcode = run(algos=args.algo,
                   scale=args.scale,
                   csv_graph_file=args.csv,
                   csv_results_file="out.csv",
                   unweighted=args.unweighted,
                   symmetric=args.symmetric_graph,
                   edgefactor=args.edgefactor,
                   benchmark_dir=args.benchmark_dir,
                   dask_scheduler_file=args.dask_scheduler_file,
                   rmm_pool_size=args.rmm_pool_size)

    sys.exit(exitcode)
