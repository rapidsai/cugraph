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


from benchmark import benchmark

import sys
def log(s, end="\n"):
    print(s, end=end)
    sys.stdout.flush()


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

        #add starting node to algos: BFS and SSSP
        for i, algo in enumerate (algo_func_param_list):
            if benchmark(algo).name in ["bfs", "sssp"]:
                param={}
                param["start"]=self.input_dataframe['src'].head()[0]
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


    def run(self):
        """
        Run and time the graph construction step, then run and time each algo.
        """
        self.results = []

        log(f"running {self.construct_graph.name}...", end="")
        result = self.construct_graph(self.input_dataframe,
                                      *self.construct_graph_func_args)
        log("done.")
        G = result.retval
        self.results.append(result)

        #algos with transposed=True : PageRank, Katz
        #algos with transposed=False: BFS, SSSP, Louvain
        for i in range(len(self.algos)):
            if self.algos[i][0].name in ["pagerank", "katz"]: #set transpose=True when renumbering
                if self.algos[i][0].name == "katz" and self.construct_graph.name == "from_dask_cudf_edgelist":
                    largest_out_degree = G.out_degree().compute().\
                    nlargest(n=1, columns="degree") #compute outdegree before renumbering because outdegree has transpose=False
                    largest_out_degree = largest_out_degree["degree"].iloc[0]
                    katz_alpha = 1 / (largest_out_degree + 1)
                    self.algos[i][1]["alpha"] = katz_alpha
                elif self.algos[i][0].name == "katz" and self.construct_graph.name == "from_cudf_edgelist":
                    largest_out_degree = G.out_degree().nlargest(n=1, columns="degree")
                    largest_out_degree = largest_out_degree["degree"].iloc[0]
                    katz_alpha = 1 / (largest_out_degree + 1)
                    self.algos[i][1]["alpha"] = katz_alpha
                if hasattr(G, "compute_renumber_edge_list"):
                    G.compute_renumber_edge_list(transposed=True)
            else: #set transpose=False when renumbering
                log("running compute_renumber_edge_list...", end="")
                if hasattr(G, "compute_renumber_edge_list"):
                    G.compute_renumber_edge_list(transposed=False)
                log("done.")
        # FIXME: need to handle individual algo args
        for ((algo, params), validator) in zip(self.algos, self.validators):
            log(f"running {algo.name} (warmup)...", end="")
            algo(G, **params)
            log("done.")
            log(f"running {algo.name}...", end="")
            result = algo(G, **params)
            log("done.")

            if validator:
                result.validator_result = validator(result.retval, G)

            self.results.append(result)
            # Reclaim memory since computed algo result is no longer needed
            result.retval = None

        return False not in [r.validator_result for r in self.results]
