import argparse
import sys
from collections import OrderedDict

from scipy.io import mmread

import cugraph
import cudf

from benchmark import (
    Benchmark, logExeTime, logGpuMetrics, printLastResult, nop
)

from asv_report import (
    cugraph_update_asv
)


# Update this function to add new algos
def getBenchmarks(G, edgelist_gdf, args):
    benches = [
        Benchmark(name="cugraph.pagerank",
                  func=cugraph.pagerank,
                  args=(G, args.damping_factor, None, args.max_iter,
                        args.tolerance)),
        Benchmark(name="cugraph.bfs",
                  func=cugraph.bfs,
                  args=(G, args.source, True)),
        Benchmark(name="cugraph.sssp",
                  func=cugraph.sssp,
                  args=(G, args.source)),
        #         extraRunWrappers=[noStdoutWrapper]),
        Benchmark(name="cugraph.jaccard",
                  func=cugraph.jaccard,
                  args=(G,)),
        Benchmark(name="cugraph.louvain",
                  func=cugraph.louvain,
                  args=(G,)),
        Benchmark(name="cugraph.weakly_connected_components",
                  func=cugraph.weakly_connected_components,
                  args=(G,)),
        Benchmark(name="cugraph.overlap",
                  func=cugraph.overlap,
                  args=(G,)),
        Benchmark(name="cugraph.triangles",
                  func=cugraph.triangles,
                  args=(G,)),
        Benchmark(name="cugraph.spectralBalancedCutClustering",
                  func=cugraph.spectralBalancedCutClustering,
                  args=(G, 2)),
        Benchmark(name="cugraph.spectralModularityMaximizationClustering",
                  func=cugraph.spectralModularityMaximizationClustering,
                  args=(G, 2)),
        Benchmark(name="cugraph.renumber",
                  func=cugraph.renumber,
                  args=(edgelist_gdf["src"], edgelist_gdf["dst"])),
        Benchmark(name="cugraph.graph.view_adj_list",
                  func=G.view_adj_list),
        Benchmark(name="cugraph.graph.degree",
                  func=G.degree),
        Benchmark(name="cugraph.graph.degrees",
                  func=G.degrees),
    ]
    # Return a dictionary of Benchmark name to Benchmark obj mappings
    return dict([(b.name, b) for b in benches])


########################################
# cugraph benchmarking utilities
def loadDataFile(file_name, csv_delimiter=' '):
    file_type = file_name.split(".")[-1]

    if file_type == "mtx":
        edgelist_gdf = read_mtx(file_name)
    elif file_type == "csv":
        edgelist_gdf = read_csv(file_name, csv_delimiter)
    else:
        raise ValueError("bad file type: '%s', %s " % (file_type, file_name) +
                         "must have a .csv or .mtx extension")
    return edgelist_gdf


def createGraph(edgelist_gdf, auto_csr):
    G = cugraph.Graph()
    G.add_edge_list(edgelist_gdf["src"], edgelist_gdf["dst"],
                    edgelist_gdf["val"])
    if auto_csr == 0:
        G.view_adj_list()
        G.view_transposed_adj_list()
    return G


def read_mtx(mtx_file):
    M = mmread(mtx_file).asfptype()
    gdf = cudf.DataFrame()
    gdf['src'] = cudf.Series(M.row)
    gdf['dst'] = cudf.Series(M.col)
    if M.data is None:
        gdf['val'] = 1.0
    else:
        gdf['val'] = cudf.Series(M.data)

    return gdf


def read_csv(csv_file, delimiter):
    cols = ["src", "dst"]
    dtypes = OrderedDict([
            ("src", "int32"),
            ("dst", "int32")
            ])

    gdf = cudf.read_csv(csv_file, names=cols, delimiter=delimiter,
                        dtype=list(dtypes.values()))
    gdf['val'] = 1.0
    if gdf['val'].null_count > 0:
        print("The reader failed to parse the input")
    if gdf['src'].null_count > 0:
        print("The reader failed to parse the input")
    if gdf['dst'].null_count > 0:
        print("The reader failed to parse the input")
    return gdf


def parseCLI(argv):
    parser = argparse.ArgumentParser(description='CuGraph benchmark script.')
    parser.add_argument('file', type=str,
                        help='Path to the input file')
    parser.add_argument('--algo', type=str, action="append",
                        help='Algorithm to run, must be one of %s, or "ALL"'
                        % ", ".join(['"%s"' % k
                                     for k in getAllPossibleAlgos()]))
    parser.add_argument('--damping_factor', type=float, default=0.85,
                        help='Damping factor for pagerank algo. Default is '
                        '0.85')
    parser.add_argument('--max_iter', type=int, default=100,
                        help='Maximum number of iteration for any iterative '
                        'algo. Default is 100')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                        help='Tolerance for any approximation algo. Default '
                        'is 1e-5')
    parser.add_argument('--source', type=int, default=0,
                        help='Source for bfs or sssp. Default is 0')
    parser.add_argument('--auto_csr', type=int, default=0,
                        help='Automatically do the csr and transposed '
                        'transformations. Default is 0, switch to another '
                        'value to enable')
    parser.add_argument('--delimiter', type=str, choices=["tab", "space"],
                        default="space",
                        help='Delimiter for csv files (default is space)')
    parser.add_argument('--update_results_dir', type=str,
                        help='Add (and compare) results to the dir specified')
    parser.add_argument('--update_asv_dir', type=str,
                        help='Add results to the specified ASV dir in ASV '
                        'format')
    parser.add_argument('--report_cuda_ver', type=str, default="",
                        help='The CUDA version to include in reports')
    parser.add_argument('--report_python_ver', type=str, default="",
                        help='The Python version to include in reports')
    parser.add_argument('--report_os_type', type=str, default="",
                        help='The OS type to include in reports')
    parser.add_argument('--report_machine_name', type=str, default="",
                        help='The machine name to include in reports')

    return parser.parse_args(argv)


def getAllPossibleAlgos():
    return list(getBenchmarks(nop, nop, nop).keys())


###############################################################################
if __name__ == "__main__":
    perfData = {}
    args = parseCLI(sys.argv[1:])

    # set algosToRun based on the command line args
    allPossibleAlgos = getAllPossibleAlgos()
    if args.algo and ("ALL" not in args.algo):
        allowedAlgoNames = allPossibleAlgos + ["ALL"]
        if (set(args.algo) - set(allowedAlgoNames)) != set():
            raise ValueError(
                "bad algo(s): '%s', must be in set of %s" %
                (args.algo, ", ".join(['"%s"' % a for a in allowedAlgoNames])))
        algosToRun = args.algo
    else:
        algosToRun = allPossibleAlgos

    # Update the various wrappers with a list to log to and formatting settings
    logExeTime.perfData = perfData
    logGpuMetrics.perfData = perfData
    printLastResult.perfData = perfData

    # Load the data file and create a Graph, treat these as benchmarks too
    csvDelim = {"space": ' ', "tab": '\t'}[args.delimiter]
    edgelist_gdf = Benchmark(loadDataFile, "cugraph.loadDataFile",
                             args=(args.file, csvDelim)).run()
    G = Benchmark(createGraph, "cugraph.createGraph",
                  args=(edgelist_gdf, args.auto_csr)).run()

    if G is None:
        raise RuntimeError("could not create graph!")

    print("-" * 80)

    benches = getBenchmarks(G, edgelist_gdf, args)

    for algo in algosToRun:
        benches[algo].run()

    # reports ########################
    if args.update_results_dir:
        raise NotImplementedError

    # import pprint
    # pprint.pprint(perfData, open("data","w"))

    if args.update_asv_dir:
        # special case: do not include the full path to the datasetName, since
        # the leading parts are redundant and take up UI space.
        datasetName = "/".join(args.file.split("/")[-3:])

        cugraph_update_asv(asvDir=args.update_asv_dir,
                           datasetName=datasetName,
                           algoRunResults=perfData,
                           cudaVer=args.report_cuda_ver,
                           pythonVer=args.report_python_ver,
                           osType=args.report_os_type,
                           machineName=args.report_machine_name)
