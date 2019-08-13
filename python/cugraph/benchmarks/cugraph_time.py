import cugraph
import cudf
import sys, os
import time
import numpy as np
from collections import OrderedDict
from scipy.io import mmread
import argparse

########################################
# Update this function to add new algos
########################################
def getAlgoData(G, args):
    """
    keyname = algo method/function name
    args = args to pass the method/function (default is no args)
    obj = object containing the method/function (default is the cugraph module)
    extraWrappers = list of functions that return a callable, used for
                    "wrapping" the algo to modify its environment, add timers,
                    log calls, etc.
    """
    algoData = {"pagerank" :
                {"args" : (G, args.damping_factor, args.max_iter, args.tolerance),
                },
                "bfs" :
                {"args" : (G, args.source, True),
                },
                "sssp" :
                {"args" : (G, args.source),
                 "extraWrappers" : [noStdoutWrapper],
                },
                "jaccard" :
                {"args" : (G,),
                },
                "louvain" :
                {"args" : (G,),
                },
                "weakly_connected_components" :
                {"args" : (G,),
                },
                "overlap" :
                {"args" : (G,),
                },
                "triangles" :
                {"args" : (G,),
                },
                "spectralBalancedCutClustering" :
                {"args" : (G, 2),
                },
                "spectralModularityMaximizationClustering" :
                {"args" : (G, 2),
                },
                "renumber" :
                {"args" : (args.source, args.source),  # FIXME: 2nd arg should be dest
                },
                "view_adj_list" :
                {"obj" : G,
                },
                "degree" :
                {"obj" : G,
                },
                "degrees" :
                {"obj" : G,
                },
    }
    return algoData


def loadDataFile(file_name, file_type, delimeter=' '):
    if file_type == "mtx" :
        edgelist_gdf = read_mtx(file_name)
    elif file_type == "csv" :
        edgelist_gdf = read_csv(file_name, delimeter)
    else:
        raise ValueError("bad file type: '%s'" % file_type)
    return edgelist_gdf


def createGraph(edgelist_gdf, auto_csr):
    G = cugraph.Graph()
    G.add_edge_list(edgelist_gdf["src"], edgelist_gdf["dst"], edgelist_gdf["val"])
    if auto_csr == 0 :
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


def read_csv(csv_file):
    cols = ["src", "dst"]
    dtypes = OrderedDict([
            ("src", "int32"),
            ("dst", "int32")
            ])
    gdf = cudf.read_csv(csv_file, names=cols, delimeter=delimeter, dtype=list(dtypes.values()) )
    gdf['val'] = 1.0
    if gdf['val'].null_count > 0 :
        print("The reader failed to parse the input")
    if gdf['src'].null_count > 0 :
        print("The reader failed to parse the input")
    if gdf['dst'].null_count > 0 :
        print("The reader failed to parse the input")
    return gdf


# wrappers for running and observing algorithms
def noStdoutWrapper(algoFunction):
    def wrapper(*algoArgs):
        prev = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            retVal = algoFunction(*algoArgs)
            sys.stdout = prev
            return retVal
        except:
            sys.stdout = prev
            raise
    return wrapper


def logExeTime(algoFunction, perfData):
    def wrapper(*algoArgs):
        retVal = None
        try:
            st = time.time()
            retVal = algoFunction(*algoArgs)
        except Exception as e:
            perfData.append((algoFunction.__name__, "ERROR: %s" % e))
            return
        perfData.append((algoFunction.__name__, (time.time()-st)))
        return retVal
    return wrapper


def parseCLI(argv):
    parser = argparse.ArgumentParser(description='CuGraph benchmark script.')
    parser.add_argument('file', type=str,
                        help='Path to the input file')
    parser.add_argument('--file_type', type=str, default="mtx", choices=["mtx", "csv"],
                        help='Input file type : csv or mtx. If csv, cuDF reader is used (set for  [src dest] pairs separated by a tab). If mtx, Scipy reder is used (slow but supports weights). Default is mtx.')
    parser.add_argument('--algo', type=str, action="append",
                        help='Algorithm to run, must be one of %s, or "all"' % ", ".join(['"%s"' % k for k in getAllPossibleAlgos()]))
    parser.add_argument('--damping_factor', type=float,default=0.85,
                        help='Damping factor for pagerank algo. Default is 0.85')
    parser.add_argument('--max_iter', type=int, default=100,
                        help='Maximum number of iteration for any iterative algo. Default is 100')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                        help='Tolerance for any approximation algo. Default is 1e-5')
    parser.add_argument('--source', type=int, default=0,
                        help='Source for bfs or sssp. Default is 0')
    parser.add_argument('--auto_csr', type=int, default=0,
                        help='Automatically do the csr and transposed transformations. Default is 0, switch to another value to enable')
    parser.add_argument('--times_only', action="store_true",
                        help='Only output the times, no table')
    parser.add_argument('--delimeter', type=str, choices=["tab", "space"], default="space",
                        help='Delimeter for csv files (default is space)')
    return parser.parse_args(argv)


def getAllPossibleAlgos():
    class fakeArgs:
        def __getattr__(self, a): return None
    return list(getAlgoData(None, fakeArgs()).keys())


################################################################################
if __name__ == "__main__":
    perfData = []
    args = parseCLI(sys.argv[1:])
    delimeter = {"space":' ', "tab":'\t'}[args.delimeter]

    allPossibleAlgos = getAllPossibleAlgos()
    if args.algo:
        if set(args.algo) != set(allPossibleAlgos):
            raise ValueError("bad algo: '%s', must be one of %s" \
                             % (args.algo, ", ".join(['"%s"' % a for a in allPossibleAlgos])))
        algosToRun = args.algo
    else:
        algosToRun = allPossibleAlgos

    # Load the data file and create a Graph, include exe time in perfData
    edgelist_gdf = logExeTime(loadDataFile, perfData)(args.file,
                                                      args.file_type,
                                                      delimeter)
    G = logExeTime(createGraph, perfData)(edgelist_gdf, args.auto_csr)

    if G is None:
        raise RuntimeError("could not create graph!")

    # Get the data on the algorithms present and how to run them
    algoData = getAlgoData(G, args)

    # For each algo to run, look up the object it belongs to (the cugraph module
    # by default), the args it needs passed (none by default), and any extra
    # function wrappers that should be applied (none by default).
    for algo in algosToRun:
        obj = algoData[algo].get("obj", cugraph)
        algoArgs = algoData[algo].get("args", ())
        extraWrappers = algoData[algo].get("extraWrappers", [])

        # get the callable, wrap it in any wrappers (which results in a wrapped
        # callable), wrap it in the logger, then finally call it with algoArgs.
        callable = getattr(obj, algo)
        for wrapper in extraWrappers:
            callable = wrapper(callable)
        callable = logExeTime(callable, perfData)
        callable(*algoArgs)

    print()
    if args.times_only:
        print(",".join([str(exeTime) for (name, exeTime) in perfData]))
    else:
        # Print a formatted table of the perfData
        nameCellWidth = max([len(name) for (name, exeTime) in perfData])
        exeTimeCellWith = max([len(str(exeTime)) for (name, exeTime) in perfData])
        for (name, exeTime) in perfData:
            print("%s | %s" % (name.ljust(nameCellWidth),
                               str(exeTime).ljust(exeTimeCellWith)))
