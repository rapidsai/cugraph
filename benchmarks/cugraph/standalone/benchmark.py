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
import time
from functools import wraps


class BenchmarkedResult:
    """
    Class to hold results (the return value of the callable being benchmarked
    and meta-data about the benchmarked function) of a benchmarked function run.
    """
    def __init__(self, name, retval, runtime, params=None):
        self.name = name
        self.retval = retval
        self.runtime = runtime
        self.params = params or {}
        self.validator_result = True


def benchmark(func):
    """
    Returns a callable/closure that wraps func with code to time the func call
    and return a BenchmarkedResult. The resulting callable takes the same
    args/kwargs as func.

    The BenchmarkedResult will have its params value assigned from the kwargs
    dictionary, but the func positional args are not captured. If a user needs
    the params captured for reporting purposes, they must use kwargs.  This is
    useful since positional args can be used for args that would not be
    meaningful in a benchmark result as a param to the benchmark.

    This can be used as a function decorator or a standalone function to wrap
    functions to benchmark.
    """
    benchmark_name = getattr(func, "benchmark_name", func.__name__)
    @wraps(func)
    def benchmark_wrapper(*func_args, **func_kwargs):
        t1 = time.perf_counter()
        retval = func(*func_args, **func_kwargs)
        t2 = time.perf_counter()
        return BenchmarkedResult(name=benchmark_name,
                                 retval=retval,
                                 runtime=(t2-t1),
                                 params=func_kwargs,
                                )

    # Assign the name to the returned callable as well for use in debug prints,
    # etc.
    benchmark_wrapper.name = benchmark_name
    return benchmark_wrapper


class BenchmarkRun:
    """
    Represents a benchmark "run", which can be executed by calling the run()
    method, and results are saved as BenchmarkedResult instances in the results
    list member.
    """
    def __init__(self,
                 input_dataframe,
                 construct_graph_func,
                 algo_func_param_list,
                 algo_validator_list=None
                ):
        self.input_dataframe = input_dataframe

        if type(construct_graph_func) is tuple:
            (construct_graph_func,
             self.construct_graph_func_args) = construct_graph_func
        else:
            self.construct_graph_func_args = None

        # Create benchmark instances for each algo/func to be timed.
        # FIXME: need to accept and save individual algo args
        self.construct_graph = benchmark(construct_graph_func)

        # add starting node to algos: BFS and SSSP
        # FIXME: Refactor BenchmarkRun __init__ because all the work
        # done below should be done elsewhere
        for i, algo in enumerate (algo_func_param_list):
            if benchmark(algo).name in ["bfs", "sssp", "uniform_neighbor_sample"]:
                param={}
                param["start"]=self.input_dataframe['src'].head()[0]
                if benchmark(algo).name in ["uniform_neighbor_sample"]:
                    start = [param.pop("start")]
                    param["start_list"] = start
                    param["fanout_vals"] = [1]
                algo_func_param_list[i]=(algo,)+(param,)

        self.algos = []
        for item in algo_func_param_list:
            if type(item) is tuple:
                (algo, params) = item
            else:
                (algo, params) = (item, {})
            self.algos.append((benchmark(algo), params))

        self.validators = algo_validator_list or [None] * len(self.algos)
        self.results = []


    @staticmethod
    def __log(s, end="\n"):
        print(s, end=end)
        sys.stdout.flush()


    def run(self):
        """
        Run and time the graph construction step, then run and time each algo.
        """
        self.results = []

        self.__log(f"running {self.construct_graph.name}...", end="")
        result = self.construct_graph(self.input_dataframe,
                                      *self.construct_graph_func_args)
        self.__log("done.")
        G = result.retval
        self.results.append(result)
        #
        # Algos with transposed=True : PageRank, Katz.
        # Algos with transposed=False: BFS, SSSP, Louvain, HITS,
        # Neighborhood_sampling.
        # Algos supporting the legacy_renum_only: HITS, Neighborhood_sampling
        #
        for i in range(len(self.algos)):
            # set transpose=True when renumbering
            if self.algos[i][0].name in ["pagerank", "katz"]:
                if self.algos[i][0].name == "katz":
                    if self.construct_graph.name == "from_dask_cudf_edgelist":
                        # compute out_degree before renumbering because out_degree
                        # has transpose=False
                        degree_max = G.degree()['degree'].max().compute()
                        katz_alpha = 1 / (degree_max)
                        self.algos[i][1]["alpha"] = katz_alpha
                    elif self.construct_graph.name == "from_cudf_edgelist":
                        degree_max = G.degree()['degree'].max()
                        katz_alpha = 1 / (degree_max)
                        self.algos[i][1]["alpha"] = katz_alpha
                    if hasattr(G, "compute_renumber_edge_list"):
                        G.compute_renumber_edge_list(
                            transposed=True, legacy_renum_only=True)
                else:
                    # FIXME: Pagerank still follows the old path. Update this once it
                    # follows the pylibcugraph/C path
                    if hasattr(G, "compute_renumber_edge_list"):
                        G.compute_renumber_edge_list(transposed=True)
            else: #set transpose=False when renumbering
                self.__log("running compute_renumber_edge_list...", end="")
                if hasattr(G, "compute_renumber_edge_list"):
                    if self.algos[i][0].name in ["wcc", "louvain"]:
                        # FIXME: Pagerank and Louvain still follow the old path.
                        # Update this once it follows the pylibcugraph/C path
                        G.compute_renumber_edge_list(transposed=False)
                    else:
                        G.compute_renumber_edge_list(
                            transposed=False, legacy_renum_only=True)
                self.__log("done.")
        # FIXME: need to handle individual algo args
        for ((algo, params), validator) in zip(self.algos, self.validators):
            self.__log(f"running {algo.name} (warmup)...", end="")
            algo(G, **params)
            self.__log("done.")
            self.__log(f"running {algo.name}...", end="")
            result = algo(G, **params)
            self.__log("done.")

            if validator:
                result.validator_result = validator(result.retval, G)

            self.results.append(result)
            # Reclaim memory since computed algo result is no longer needed
            result.retval = None

        return False not in [r.validator_result for r in self.results]
