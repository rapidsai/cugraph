from collections import OrderedDict
import pytest

import cudf
import cugraph

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
DATASETS = [
    "../datasets/csv/undirected/hollywood.csv",
    "../datasets/csv/undirected/europe_osm.csv",
#    "../datasets/csv/undirected/soc-twitter-2010.csv",
]


###############################################################################
# Fixtures
#
# Executed automatically when specified on a test/benchmark, and the return
# value is made available to the test/benchmark for use. Fixtures can use other
# fixtures to effectively chain their execution.
#
# For benchmarks, the operations performed in fixtures are not measured as part
# of the benchmark.
@pytest.fixture(scope="module",
                params=DATASETS)
def edgelistCreated(request):
    """
    Returns a new edgelist created from a CSV, which is specified as part of
    the parameterization for this fixture.
    """
    # FIXME: make a helper to do this and explain why it needs to be done
    # FIXME: this works here but needs to be updated to automatically create and append
    request.keywords.setdefault("fixture_param_names", dict())[request.fixturename] = ["dataset"]
    return getEdgelistFromCsv(request.param)


@pytest.fixture(scope="module")
def graphCreated(edgelistCreated):
    """
    Returns a new Graph object created from the return value of the
    edgelistCreated fixture.
    """
    return getGraphFromEdgelist(edgelistCreated)


###############################################################################
# Benchmarks
@pytest.mark.benchmark(group="ETL")
@pytest.mark.parametrize("csvFileName", DATASETS)
def bench_create_edgelist(gpubenchmark, csvFileName):
    gpubenchmark(getEdgelistFromCsv, csvFileName)


@pytest.mark.benchmark(group="ETL")
def bench_create_graph(gpubenchmark, edgelistCreated):
    gpubenchmark(getGraphFromEdgelist, edgelistCreated, False, False, False)


# def bench_pagerank(gpubenchmark, graphCreated):
#     gpubenchmark(cugraph.pagerank, graphCreated, damping_factor=0.85, None, max_iter=100, tolerance=1e-5)

def bench_bfs(gpubenchmark, graphCreated):
    gpubenchmark(cugraph.bfs, graphCreated, 0)


def bench_sssp(gpubenchmark, graphCreated):
    gpubenchmark(cugraph.sssp, graphCreated, 0)


def bench_jaccard(gpubenchmark, graphCreated):
    gpubenchmark(cugraph.jaccard, graphCreated)


def bench_louvain(gpubenchmark, graphCreated):
    gpubenchmark(cugraph.louvain, graphCreated)


def bench_weakly_connected_components(gpubenchmark, graphCreated):
    gpubenchmark(cugraph.weakly_connected_components, graphCreated)


def bench_overlap(gpubenchmark, graphCreated):
    gpubenchmark(cugraph.overlap, graphCreated)


def bench_triangles(gpubenchmark, graphCreated):
    gpubenchmark(cugraph.triangles, graphCreated)


def bench_spectralBalancedCutClustering(gpubenchmark, graphCreated):
    gpubenchmark(cugraph.spectralBalancedCutClustering, graphCreated, 2)


# def bench_spectralModularityMaximizationClustering(gpubenchmark, graphCreated):
#     gpubenchmark(cugraph.spectralModularityMaximizationClustering, graphCreated, 2)


# def bench_renumber(gpubenchmark, edgelistCreated):
#     gpubenchmark(cugraph.renumber, edgelistCreated["src"], edgelistCreated["dst"])


def bench_graph_degree(gpubenchmark, graphCreated):
    gpubenchmark(graphCreated.degree)


def bench_graph_degrees(gpubenchmark, graphCreated):
    gpubenchmark(graphCreated.degrees)
