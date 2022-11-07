# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
import sys
from collections import OrderedDict

from scipy.io import mmread

import cugraph
import cudf

from benchmark import Benchmark


###############################################################################
# Update this function to add new algos
###############################################################################
def getBenchmarks(G, edgelist_gdf, args):
    """Returns a dictionary of benchmark name to Benchmark objs. This dictionary
    is used when processing the command-line args to this script so the script
    can run a specific benchmakr by name.

    The "edgelist_gdf" and "args" args are common to many benchmark runs, and
    provided to this function to make it easy to pass to individual Benchmark
    objs.  The "args" arg in particular is a dictionary built from processing
    the command line args to this script, and allow special parameters to be
    added to the command line for use by specific benchmarks.

    To add a new benchmark to run, simply add an instance of a Benchmark to the
    "benches" list below.

     * The Benchmark instance ctor takes 3 args:
       * "name" - the name of the benchmark which will show up in reports,
                  output, etc.
       * "func" - the function object which the benchmark will call.  This can
                  be any callable.
       * "args" - a tuple of args that are to be passed to the func callable.

    A Benchmark object will, by default, run the callable with the args
    provided as-is, and log the execution time and various GPU metrics.  The
    callable provided is written independent of the benchmarking code (this is
    good for separation of concerns, bad if you need to do a custom
    measurement).

    If a new benchmark needs a special command-line parameter, add a new flag
    to the command-line processing function and access it via the "args"
    dictionary when passing args to the Benchmark ctor.

    """

    benches = [
        Benchmark(
            name="cugraph.pagerank",
            func=cugraph.pagerank,
            args=(G, args.damping_factor, None, args.max_iter, args.tolerance),
        ),
        Benchmark(name="cugraph.bfs", func=cugraph.bfs, args=(G, args.source, True)),
        Benchmark(name="cugraph.sssp", func=cugraph.sssp, args=(G, args.source)),
        Benchmark(name="cugraph.jaccard", func=cugraph.jaccard, args=(G,)),
        Benchmark(name="cugraph.louvain", func=cugraph.louvain, args=(G,)),
        Benchmark(
            name="cugraph.weakly_connected_components",
            func=cugraph.weakly_connected_components,
            args=(G,),
        ),
        Benchmark(name="cugraph.overlap", func=cugraph.overlap, args=(G,)),
        Benchmark(name="cugraph.triangles", func=cugraph.triangles, args=(G,)),
        Benchmark(
            name="cugraph.spectralBalancedCutClustering",
            func=cugraph.spectralBalancedCutClustering,
            args=(G, 2),
        ),
        Benchmark(
            name="cugraph.spectralModularityMaximizationClustering",
            func=cugraph.spectralModularityMaximizationClustering,
            args=(G, 2),
        ),
        Benchmark(
            name="cugraph.renumber",
            func=cugraph.renumber,
            args=(edgelist_gdf["src"], edgelist_gdf["dst"]),
        ),
        Benchmark(name="cugraph.graph.degree", func=G.degree),
        Benchmark(name="cugraph.graph.degrees", func=G.degrees),
    ]
    # Return a dictionary of Benchmark name to Benchmark obj mappings
    return dict([(b.name, b) for b in benches])


########################################
# cugraph benchmarking utilities
def loadDataFile(file_name, csv_delimiter=" "):
    file_type = file_name.split(".")[-1]

    if file_type == "mtx":
        edgelist_gdf = read_mtx(file_name)
    elif file_type == "csv":
        edgelist_gdf = read_csv(file_name, csv_delimiter)
    else:
        raise ValueError(
            "bad file type: '%s', %s " % (file_type, file_name)
            + "must have a .csv or .mtx extension"
        )
    return edgelist_gdf


def createGraph(edgelist_gdf, createDiGraph, renumber, symmetrized):
    if createDiGraph:
        G = cugraph.DiGraph()
    else:
        G = cugraph.Graph(symmetrized=symmetrized)
    G.from_cudf_edgelist(
        edgelist_gdf,
        source="src",
        destination="dst",
        edge_attr="val",
        renumber=renumber,
    )
    return G


def computeAdjList(graphObj, transposed=False):
    """
    Compute the adjacency list (or transposed adjacency list if transposed is
    True) on the graph obj. This can be run as a benchmark itself, and is often
    run separately so adj list computation isn't factored into an algo
    benchmark.
    """
    if transposed:
        G.view_transposed_adj_list()
    else:
        G.view_adj_list()


def read_mtx(mtx_file):
    M = mmread(mtx_file).asfptype()
    gdf = cudf.DataFrame()
    gdf["src"] = cudf.Series(M.row)
    gdf["dst"] = cudf.Series(M.col)
    if M.data is None:
        gdf["val"] = 1.0
    else:
        gdf["val"] = cudf.Series(M.data)

    return gdf


def read_csv(csv_file, delimiter):
    cols = ["src", "dst", "val"]
    dtypes = OrderedDict(
        [
            ("src", "int32"),
            ("dst", "int32"),
            ("val", "float32"),
        ]
    )

    gdf = cudf.read_csv(
        csv_file, names=cols, delimiter=delimiter, dtype=list(dtypes.values())
    )

    if gdf["src"].null_count > 0:
        print("The reader failed to parse the input")
    if gdf["dst"].null_count > 0:
        print("The reader failed to parse the input")
    # Assume an edge weight of 1.0 if dataset does not provide it
    if gdf["val"].null_count > 0:
        gdf["val"] = 1.0
    return gdf


def parseCLI(argv):
    parser = argparse.ArgumentParser(description="CuGraph benchmark script.")
    parser.add_argument("file", type=str, help="Path to the input file")
    parser.add_argument(
        "--algo",
        type=str,
        action="append",
        help='Algorithm to run, must be one of %s, or "ALL"'
        % ", ".join(['"%s"' % k for k in getAllPossibleAlgos()]),
    )
    parser.add_argument(
        "--damping_factor",
        type=float,
        default=0.85,
        help="Damping factor for pagerank algo. Default is " "0.85",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum number of iteration for any iterative " "algo. Default is 100",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Tolerance for any approximation algo. Default " "is 1e-5",
    )
    parser.add_argument(
        "--source", type=int, default=0, help="Source for bfs or sssp. Default is 0"
    )
    parser.add_argument(
        "--compute_adj_list",
        action="store_true",
        help="Compute and benchmark the adjacency list "
        "computation separately. Default is to NOT compute "
        "the adjacency list and allow the algo to compute it "
        "if necessary.",
    )
    parser.add_argument(
        "--compute_transposed_adj_list",
        action="store_true",
        help="Compute and benchmark the transposed adjacency "
        "list computation separately. Default is to NOT "
        "compute the transposed adjacency list and allow the "
        "algo to compute it if necessary.",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        choices=["tab", "space"],
        default="space",
        help="Delimiter for csv files (default is space)",
    )
    parser.add_argument(
        "--update_results_dir",
        type=str,
        help="Add (and compare) results to the dir specified",
    )
    parser.add_argument(
        "--update_asv_dir",
        type=str,
        help="Add results to the specified ASV dir in ASV " "format",
    )
    parser.add_argument(
        "--report_cuda_ver",
        type=str,
        default="",
        help="The CUDA version to include in reports",
    )
    parser.add_argument(
        "--report_python_ver",
        type=str,
        default="",
        help="The Python version to include in reports",
    )
    parser.add_argument(
        "--report_os_type",
        type=str,
        default="",
        help="The OS type to include in reports",
    )
    parser.add_argument(
        "--report_machine_name",
        type=str,
        default="",
        help="The machine name to include in reports",
    )
    parser.add_argument(
        "--digraph",
        action="store_true",
        help="Create a directed graph (default is undirected)",
    )

    return parser.parse_args(argv)


def getAllPossibleAlgos():
    # Use the getBenchmarks() function to generate a list of benchmark names
    # from the keys of the dictionary getBenchmarks() returns.  Use a "nop"
    # object since getBenchmarks() will try to access attrs for the args passed
    # in, and there's no point in keeping track of the actual objects needed
    # here since all this needs is the keys (not the values).
    class Nop:
        def __getattr__(self, attr):
            return Nop()

        def __getitem__(self, key):
            return Nop()

        def __call__(self, *args, **kwargs):
            return Nop()

    nop = Nop()

    return list(getBenchmarks(nop, nop, nop).keys())


###############################################################################
if __name__ == "__main__":
    args = parseCLI(sys.argv[1:])

    # set algosToRun based on the command line args
    allPossibleAlgos = getAllPossibleAlgos()
    if args.algo and ("ALL" not in args.algo):
        allowedAlgoNames = allPossibleAlgos + ["ALL"]
        if (set(args.algo) - set(allowedAlgoNames)) != set():
            raise ValueError(
                "bad algo(s): '%s', must be in set of %s"
                % (args.algo, ", ".join(['"%s"' % a for a in allowedAlgoNames]))
            )
        algosToRun = args.algo
    else:
        algosToRun = allPossibleAlgos

    # Load the data file and create a Graph, treat these as benchmarks too. The
    # Benchmark run() method returns the result of the function being
    # benchmarked. In this case, "loadDataFile" and "createGraph" return a
    # Dataframe and Graph object respectively, so save those and use them for
    # future benchmarks.
    csvDelim = {"space": " ", "tab": "\t"}[args.delimiter]
    edgelist_gdf = Benchmark(
        loadDataFile, "cugraph.loadDataFile", args=(args.file, csvDelim)
    ).run()
    renumber = True
    symmetrized = True

    G = Benchmark(
        createGraph,
        "cugraph.createGraph",
        args=(edgelist_gdf, args.digraph, renumber, symmetrized),
    ).run()

    if G is None:
        raise RuntimeError("could not create graph!")

    # compute the adjacency list upfront as a separate benchmark. Special case:
    # if pagerank is being benchmarked and the transposed adj matrix is
    # requested, compute that too or instead. It's recommended that a pagerank
    # benchmark be performed in a separate run since there's only one Graph obj
    # and both an adj list and transposed adj list are probably not needed.
    if args.compute_adj_list:
        Benchmark(computeAdjList, "cugraph.graph.view_adj_list", args=(G, False)).run()
    if args.compute_transposed_adj_list and ("cugraph.pagerank" in algosToRun):
        Benchmark(
            computeAdjList, "cugraph.graph.view_transposed_adj_list", args=(G, True)
        ).run()

    print("-" * 80)

    # get the individual benchmark functions and run them
    benches = getBenchmarks(G, edgelist_gdf, args)
    for algo in algosToRun:
        benches[algo].run(n=3)  # mean of 3 runs

    # reports ########################
    if args.update_results_dir:
        raise NotImplementedError

    if args.update_asv_dir:
        # import this here since it pulls in a 3rd party package (asvdb) which
        # may not be appreciated by non-ASV users.
        from asv_report import cugraph_update_asv

        # special case: do not include the full path to the datasetName, since
        # the leading parts are redundant and take up UI space.
        datasetName = "/".join(args.file.split("/")[-3:])

        cugraph_update_asv(
            asvDir=args.update_asv_dir,
            datasetName=datasetName,
            algoRunResults=Benchmark.resultsDict,
            cudaVer=args.report_cuda_ver,
            pythonVer=args.report_python_ver,
            osType=args.report_os_type,
            machineName=args.report_machine_name,
        )
