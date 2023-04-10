# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
from cugraph.structure.number_map import NumberMap
from cugraph.testing import utils
from pylibcugraph.testing import gen_fixture_params_product
from cugraph.utilities.utils import is_device_version_less_than
import rmm

from cugraph_benchmarking.params import (
    directed_datasets,
    undirected_datasets,
    managed_memory,
    pool_allocator,
)

fixture_params = gen_fixture_params_product(
    (directed_datasets + undirected_datasets, "ds"),
    (managed_memory, "mm"),
    (pool_allocator, "pa"))

###############################################################################
# Helpers
def createGraph(csvFileName, graphType=None):
    """
    Helper function to create a Graph (directed or undirected) based on
    csvFileName.
    """
    if graphType is None:
        # There's potential value in verifying that a directed graph can be
        # created from a undirected dataset, and an undirected from a directed
        # dataset. (For now?) do not include those combinations to keep
        # benchmark runtime and complexity lower, and assume tests have
        # coverage to verify correctness for those combinations.
        if "directed" in csvFileName.parts:
            graphType = cugraph.Graph(directed=True)
        else:
            graphType = cugraph.Graph()

    return cugraph.from_cudf_edgelist(
        utils.read_csv_file(csvFileName),
        source="0", destination="1", edge_attr="2",
        create_using=graphType,
        renumber=True)


# Record the current RMM settings so reinitialize() will be called only when a
# change is needed (RMM defaults both values to False). This allows the
# --no-rmm-reinit option to prevent reinitialize() from being called at all
# (see conftest.py for details).
RMM_SETTINGS = {"managed_mem": False,
                "pool_alloc": False}


def reinitRMM(managed_mem, pool_alloc):

    if (managed_mem != RMM_SETTINGS["managed_mem"]) or \
       (pool_alloc != RMM_SETTINGS["pool_alloc"]):

        rmm.reinitialize(
            managed_memory=managed_mem,
            pool_allocator=pool_alloc,
            initial_pool_size=2 << 27
        )
        RMM_SETTINGS.update(managed_mem=managed_mem,
                            pool_alloc=pool_alloc)


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
                params=fixture_params)
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
    reinitRMM(request.param[1], request.param[2])
    return utils.read_csv_file(csvFileName)


@pytest.fixture(scope="module",
                params=fixture_params)
def graphWithAdjListComputed(request):
    """
    Create a Graph obj from the CSV file in param, compute the adjacency list
    and return it.
    """
    setFixtureParamNames(request, ["dataset", "managed_mem", "pool_allocator"])
    csvFileName = request.param[0]
    reinitRMM(request.param[1], request.param[2])

    G = createGraph(csvFileName, cugraph.structure.graph_classes.Graph)
    G.view_adj_list()
    return G


@pytest.fixture(scope="module",
                params=fixture_params)
def anyGraphWithAdjListComputed(request):
    """
    Create a Graph (directed or undirected) obj based on the param, compute the
    adjacency list and return it.
    """
    setFixtureParamNames(request, ["dataset", "managed_mem", "pool_allocator"])
    csvFileName = request.param[0]
    reinitRMM(request.param[1], request.param[2])

    G = createGraph(csvFileName)
    G.view_adj_list()
    return G


@pytest.fixture(scope="module",
                params=fixture_params)
def anyGraphWithTransposedAdjListComputed(request):
    """
    Create a Graph (directed or undirected) obj based on the param, compute the
    transposed adjacency list and return it.
    """
    setFixtureParamNames(request, ["dataset", "managed_mem", "pool_allocator"])
    csvFileName = request.param[0]
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
                 create_using=cugraph.structure.graph_classes.Graph,
                 renumber=False)


# Creating directed Graphs on small datasets runs in micro-seconds, which
# results in thousands of rounds before the default threshold is met, so lower
# the max_time for this benchmark.
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
                 create_using=cugraph.Graph(directed=True),
                 renumber=False)


@pytest.mark.ETL
def bench_renumber(gpubenchmark, edgelistCreated):
    gpubenchmark(NumberMap.renumber, edgelistCreated, "0", "1")


def bench_pagerank(gpubenchmark, anyGraphWithTransposedAdjListComputed):
    gpubenchmark(cugraph.pagerank, anyGraphWithTransposedAdjListComputed)


def bench_bfs(gpubenchmark, anyGraphWithAdjListComputed):
    gpubenchmark(cugraph.bfs, anyGraphWithAdjListComputed, 0)


def bench_force_atlas2(gpubenchmark, anyGraphWithAdjListComputed):
    gpubenchmark(cugraph.force_atlas2, anyGraphWithAdjListComputed,
                 max_iter=50)


def bench_sssp(gpubenchmark, anyGraphWithAdjListComputed):
    gpubenchmark(cugraph.sssp, anyGraphWithAdjListComputed, 0)


def bench_jaccard(gpubenchmark, graphWithAdjListComputed):
    gpubenchmark(cugraph.jaccard, graphWithAdjListComputed)


@pytest.mark.skipif(
    is_device_version_less_than((7, 0)), reason="Not supported on Pascal")
def bench_louvain(gpubenchmark, graphWithAdjListComputed):
    gpubenchmark(cugraph.louvain, graphWithAdjListComputed)


def bench_weakly_connected_components(gpubenchmark,
                                      anyGraphWithAdjListComputed):
    gpubenchmark(cugraph.weakly_connected_components,
                 anyGraphWithAdjListComputed)


def bench_overlap(gpubenchmark, anyGraphWithAdjListComputed):
    gpubenchmark(cugraph.overlap, anyGraphWithAdjListComputed)


def bench_triangle_count(gpubenchmark, graphWithAdjListComputed):
    gpubenchmark(cugraph.triangle_count, graphWithAdjListComputed)


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


def bench_edge_betweenness_centrality(gpubenchmark,
                                      anyGraphWithAdjListComputed):
    gpubenchmark(cugraph.edge_betweenness_centrality,
                 anyGraphWithAdjListComputed, k=10, seed=123)
