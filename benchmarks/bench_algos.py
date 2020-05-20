import pytest

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

import cugraph
from cugraph.tests import utils
import rmm

from .params import FIXTURE_PARAMS


###############################################################################
# Helpers
def createGraph(csvFileName, graphType=None):
    """
    Helper function to create a Graph or DiGraph based on csvFileName.
    """
    if graphType is None:
        # There's potential value in verifying that a DiGraph can be created
        # from a undirected dataset, and a Graph from a directed. (For now?) do
        # not include those combinations to keep benchmark runtime and
        # complexity lower, and assume tests have coverage to verify
        # correctness for those combinations.
        if "/directed/" in csvFileName:
            graphType = cugraph.structure.graph.DiGraph
        else:
            graphType = cugraph.structure.graph.Graph

    return cugraph.from_cudf_edgelist(
        utils.read_csv_file(csvFileName),
        source="0", destination="1",
        create_using=graphType,
        renumber=True)


def reinitRMM(managed_mem, pool_alloc):
    rmm.reinitialize(
        managed_memory=managed_mem,
        pool_allocator=pool_alloc,
        initial_pool_size=2 << 27
    )


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
                params=FIXTURE_PARAMS)
def edgelistCreated(request):
    """
    Returns a new edgelist created from a CSV, which is specified as part of
    the parameterization for this fixture.
    """
    # Since parameterized fixtures do not assign param names to param values,
    # manually call the helper to do so. Ensure the order of the name list
    # passed to it matches if there are >1 params.
    # If the request only contains n params, only the first n names are set.
    setFixtureParamNames(request, ["dataset", "managed_mem", "pool_allocator"])

    csvFileName = request.param[0]
    if len(request.param) > 1:
        reinitRMM(request.param[1], request.param[2])
    return utils.read_csv_file(csvFileName)


@pytest.fixture(scope="module",
                params=FIXTURE_PARAMS)
def graphWithAdjListComputed(request):
    """
    Create a Graph obj from the CSV file in param, compute the adjacency list
    and return it.
    """
    setFixtureParamNames(request, ["dataset", "managed_mem", "pool_allocator"])
    csvFileName = request.param[0]
    if len(request.param) > 1:
        reinitRMM(request.param[1], request.param[2])

    G = createGraph(csvFileName, cugraph.structure.graph.Graph)
    G.view_adj_list()
    return G


@pytest.fixture(scope="module",
                params=FIXTURE_PARAMS)
def anyGraphWithAdjListComputed(request):
    """
    Create a Graph (or DiGraph) obj based on the param, compute the adjacency
    list and return it.
    """
    setFixtureParamNames(request, ["dataset", "managed_mem", "pool_allocator"])
    csvFileName = request.param[0]
    if len(request.param) > 1:
        reinitRMM(request.param[1], request.param[2])

    G = createGraph(csvFileName)
    G.view_adj_list()
    return G


@pytest.fixture(scope="module",
                params=FIXTURE_PARAMS)
def anyGraphWithTransposedAdjListComputed(request):
    """
    Create a Graph (or DiGraph) obj based on the param, compute the transposed
    adjacency list and return it.
    """
    setFixtureParamNames(request, ["dataset", "managed_mem", "pool_allocator"])
    csvFileName = request.param[0]
    if len(request.param) > 1:
        reinitRMM(request.param[1], request.param[2])

    G = createGraph(csvFileName)
    G.view_transposed_adj_list()
    return G


###############################################################################
# Benchmarks
@pytest.mark.ETL
def bench_create_graph(gpubenchmark, edgelistCreated):
    gpubenchmark(cugraph.from_cudf_edgelist,
                 edgelistCreated,
                 source="0", destination="1",
                 create_using=cugraph.structure.graph.Graph,
                 renumber=False)


# Creating DiGraphs on small datasets runs in micro-seconds, which results in
# thousands of rounds before the default threshold is met, so lower the
# max_time for this benchmark.
@pytest.mark.ETL
@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=10,
    max_time=0.005
)
def bench_create_digraph(gpubenchmark, edgelistCreated):
    gpubenchmark(cugraph.from_cudf_edgelist,
                 edgelistCreated,
                 source="0", destination="1",
                 create_using=cugraph.structure.graph.DiGraph,
                 renumber=False)


@pytest.mark.ETL
def bench_renumber(gpubenchmark, edgelistCreated):
    gpubenchmark(cugraph.renumber,
                 edgelistCreated["0"],  # src
                 edgelistCreated["1"])  # dst


def bench_pagerank(gpubenchmark, anyGraphWithTransposedAdjListComputed):
    gpubenchmark(cugraph.pagerank, anyGraphWithTransposedAdjListComputed)


def bench_bfs(gpubenchmark, anyGraphWithAdjListComputed):
    gpubenchmark(cugraph.bfs, anyGraphWithAdjListComputed, 0)


def bench_sssp(gpubenchmark, anyGraphWithAdjListComputed):
    gpubenchmark(cugraph.sssp, anyGraphWithAdjListComputed, 0)


def bench_jaccard(gpubenchmark, graphWithAdjListComputed):
    gpubenchmark(cugraph.jaccard, graphWithAdjListComputed)


def bench_louvain(gpubenchmark, graphWithAdjListComputed):
    gpubenchmark(cugraph.louvain, graphWithAdjListComputed)


def bench_weakly_connected_components(gpubenchmark,
                                      anyGraphWithAdjListComputed):
    gpubenchmark(cugraph.weakly_connected_components,
                 anyGraphWithAdjListComputed)


def bench_overlap(gpubenchmark, anyGraphWithAdjListComputed):
    gpubenchmark(cugraph.overlap, anyGraphWithAdjListComputed)


def bench_triangles(gpubenchmark, graphWithAdjListComputed):
    gpubenchmark(cugraph.triangles, graphWithAdjListComputed)


def bench_spectralBalancedCutClustering(gpubenchmark,
                                        graphWithAdjListComputed):
    gpubenchmark(cugraph.spectralBalancedCutClustering,
                 graphWithAdjListComputed, 2)


@pytest.mark.skip(reason="Need to guarantee graph has weights, "
                         "not doing that yet")
def bench_spectralModularityMaximizationClustering(
        gpubenchmark, anyGraphWithAdjListComputed):
    gpubenchmark(cugraph.spectralModularityMaximizationClustering,
                 anyGraphWithAdjListComputed, 2)


def bench_graph_degree(gpubenchmark, anyGraphWithAdjListComputed):
    gpubenchmark(anyGraphWithAdjListComputed.degree)


def bench_graph_degrees(gpubenchmark, anyGraphWithAdjListComputed):
    gpubenchmark(anyGraphWithAdjListComputed.degrees)


def bench_betweenness_centrality(gpubenchmark, anyGraphWithAdjListComputed):
    gpubenchmark(cugraph.betweenness_centrality,
                 anyGraphWithAdjListComputed, k=10, seed=123)
