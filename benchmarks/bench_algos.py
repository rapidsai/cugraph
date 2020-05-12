from collections import OrderedDict
import pytest

import cudf
import cugraph

import pytest_benchmark
# FIXME: Remove this when rapids_pytest_benchmark.gpubenchmark is available
# everywhere
try:
    from rapids_pytest_benchmark import setFixtureParamNames
except ImportError:
    print("\n\nWARNING: rapids_pytest_benchmark is not installed, "
          "falling back to pytest_benchmark fixtures.\n")

    # if rapids_pytest_benchmark is not available, just perfrom time-only
    # benchmarking and replace the util functions with nops
    gpubenchmark = pytest_benchmark.plugin.benchmark

    def setFixtureParamNames(*args, **kwargs):
        pass


###############################################################################
# Utilities
#
# FIXME: move utilities to a more reusable location/module
def getEdgelistFromCsv(csvFileName, delim=' '):
    """
    Returns a cuDF DataFrame containing the columns read in from
    csvFileName. Optional delim string defaults to ' ' (space) for CSV reading.
    """
    cols = ["src", "dst", "val"]
    dtypes = OrderedDict([
            ("src", "int32"),
            ("dst", "int32"),
            ("val", "float32"),
            ])

    gdf = cudf.read_csv(csvFileName, names=cols, delimiter=delim,
                        dtype=list(dtypes.values()))

    if gdf['src'].null_count > 0:
        raise RuntimeError("The reader failed to parse the input")
    if gdf['dst'].null_count > 0:
        raise RuntimeError("The reader failed to parse the input")
    # Assume an edge weight of 1.0 if dataset does not provide it
    if gdf['val'].null_count > 0:
        gdf['val'] = 1.0
    return gdf


def getGraphFromEdgelist(edgelistGdf, createDiGraph=False,
                         renumber=False, symmetrized=False):
    """
    Returns a cugraph Graph or DiGraph object from edgelistGdf. renumber and
    symmetrized can be set to True to perform those operation on construction.
    """
    if createDiGraph:
        G = cugraph.DiGraph()
    else:
        G = cugraph.Graph(symmetrized=symmetrized)
    G.from_cudf_edgelist(edgelistGdf, source="src",
                         destination="dst", edge_attr="val",
                         renumber=renumber)
    return G


# FIXME: write and use mechanism described here for specifying datasets:
#        https://docs.rapids.ai/maintainers/datasets
# FIXME: rlr: soc-twitter-2010.csv crashes with OOM error on my HP-Z8!
UNDIRECTED_DATASETS = [
    pytest.param("../datasets/csv/undirected/hollywood.csv",
                 marks=[pytest.mark.small, pytest.mark.undirected]),
    pytest.param("../datasets/csv/undirected/europe_osm.csv",
                 marks=[pytest.mark.undirected]),
    # pytest.param("../datasets/csv/undirected/soc-twitter-2010.csv",
    #              marks=[pytest.mark.undirected]),
]
DIRECTED_DATASETS = [
    pytest.param("../datasets/csv/directed/cit-Patents.csv",
                 marks=[pytest.mark.small, pytest.mark.directed]),
    pytest.param("../datasets/csv/directed/soc-LiveJournal1.csv",
                 marks=[pytest.mark.directed]),
]

###############################################################################
# Fixtures
#
# Executed automatically when specified on a test/benchmark, and the return
# value is made available to the test/benchmark for use. Fixtures can use other
# fixtures to chain their execution.
#
# For benchmarks, the operations performed in fixtures are not measured as part
# of the benchmark.
@pytest.fixture(scope="module",
                params=UNDIRECTED_DATASETS + DIRECTED_DATASETS)
def edgelistCreated(request):
    """
    Returns a new edgelist created from a CSV, which is specified as part of
    the parameterization for this fixture.
    """
    # Since parameterized fixtures do not assign param names to param values,
    # manually call the helper to do so. Ensure the order of the name list
    # passed to it matches if there are >1 params.
    setFixtureParamNames(request, ["dataset"])
    return getEdgelistFromCsv(request.param)


@pytest.fixture(scope="module",
                params=UNDIRECTED_DATASETS)
def graphCreated(request):
    """
    Returns a new Graph object created from the return value of the
    edgelistCreated fixture.
    """
    setFixtureParamNames(request, ["dataset"])

    return getGraphFromEdgelist(getEdgelistFromCsv(request.param),
                                createDiGraph=False,
                                renumber=True,
                                symmetrized=False)

@pytest.fixture(scope="module",
                params=DIRECTED_DATASETS)
def diGraphCreated(request):
    """
    Returns a new DiGraph object created from the return value of the
    edgelistCreated fixture.
    """
    setFixtureParamNames(request, ["dataset"])

    return getGraphFromEdgelist(getEdgelistFromCsv(request.param),
                                createDiGraph=True,
                                renumber=True,
                                symmetrized=False)


@pytest.fixture(scope="module",
                params=UNDIRECTED_DATASETS + DIRECTED_DATASETS)
def anyGraphCreated(request):
    """
    Returns a new DiGraph object created from the return value of the
    edgelistCreated fixture.
    """
    setFixtureParamNames(request, ["dataset"])

    isDiGraph = "/directed/" in request.param
    return getGraphFromEdgelist(getEdgelistFromCsv(request.param),
                                createDiGraph=isDiGraph,
                                renumber=True,
                                symmetrized=False)


@pytest.fixture(scope="module")
def computeAdjList(graphCreated):
    """
    Compute the adjacency list on the graph obj.
    """
    graphCreated.view_adj_list()
    return graphCreated


@pytest.fixture(scope="module")
def computeTransposedAdjList(graphCreated):
    """
    Compute the transposed adjacency list on the graph obj.
    """
    graphCreated.view_transposed_adj_list()
    return graphCreated


###############################################################################
# Benchmarks
@pytest.mark.ETL
@pytest.mark.benchmark(group="ETL")
@pytest.mark.parametrize("dataset", UNDIRECTED_DATASETS + DIRECTED_DATASETS)
def bench_create_edgelist(gpubenchmark, dataset):
    gpubenchmark(getEdgelistFromCsv, dataset)


@pytest.mark.ETL
@pytest.mark.benchmark(group="ETL")
def bench_create_graph(gpubenchmark, edgelistCreated):
    gpubenchmark(getGraphFromEdgelist, edgelistCreated,
                 createDiGraph=False,
                 renumber=False,
                 symmetrized=False)


def bench_pagerank(gpubenchmark, anyGraphCreated):
    gpubenchmark(cugraph.pagerank, anyGraphCreated)


def bench_bfs(gpubenchmark, anyGraphCreated):
    gpubenchmark(cugraph.bfs, anyGraphCreated, 0)


def bench_sssp(gpubenchmark, anyGraphCreated):
    gpubenchmark(cugraph.sssp, anyGraphCreated, 0)


def bench_jaccard(gpubenchmark, graphCreated):
    gpubenchmark(cugraph.jaccard, graphCreated)


def bench_louvain(gpubenchmark, graphCreated):
    gpubenchmark(cugraph.louvain, graphCreated)


def bench_weakly_connected_components(gpubenchmark, anyGraphCreated):
    gpubenchmark(cugraph.weakly_connected_components, anyGraphCreated)


def bench_overlap(gpubenchmark, anyGraphCreated):
    gpubenchmark(cugraph.overlap, anyGraphCreated)


def bench_triangles(gpubenchmark, graphCreated):
    gpubenchmark(cugraph.triangles, graphCreated)


def bench_spectralBalancedCutClustering(gpubenchmark, graphCreated):
    gpubenchmark(cugraph.spectralBalancedCutClustering, graphCreated, 2)


# def bench_spectralModularityMaximizationClustering(gpubenchmark,
#                                                    anyGraphCreated):
#     gpubenchmark(cugraph.spectralModularityMaximizationClustering,
#                  anyGraphCreated, 2)


# def bench_renumber(gpubenchmark, edgelistCreated):
#     gpubenchmark(cugraph.renumber, edgelistCreated["src"],
#                  edgelistCreated["dst"])


def bench_graph_degree(gpubenchmark, anyGraphCreated):
    gpubenchmark(anyGraphCreated.degree)


def bench_graph_degrees(gpubenchmark, anyGraphCreated):
    gpubenchmark(anyGraphCreated.degrees)
