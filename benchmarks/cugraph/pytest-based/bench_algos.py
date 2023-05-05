# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

import rmm
from pylibcugraph.testing import gen_fixture_params_product

import cugraph
from cugraph.structure.number_map import NumberMap
from cugraph.generators import rmat
from cugraph.testing import utils
from cugraph.utilities.utils import is_device_version_less_than

from cugraph_benchmarking.params import (
    directed_datasets,
    undirected_datasets,
    managed_memory,
    pool_allocator,
)

# duck-type compatible Dataset for RMAT data
class RmatDataset:
    def __init__(self, scale=4, edgefactor=8):
        self.scale = scale
        self.edgefactor = edgefactor
        self._df = None
        self._graph = None

    def unload(self):
        self._df = None
        self._graph = None

    def __str__(self):
        return f"rmat_{self.scale}_{self.edgefactor}"

    def get_edgelist(self, fetch=False):
        if self._df is None:
            self._df = rmat(
                self.scale,
                (2**self.scale)*self.edgefactor,
                0.57,  # from Graph500
                0.19,  # from Graph500
                0.19,  # from Graph500
                seed or 42,
                clip_and_flip=False,
                scramble_vertex_ids=True,
                create_using=None,  # return edgelist instead of Graph instance
                mg=False
            )
            rng = np.random.default_rng(seed)
            self._df["weight"] = rng.random(size=len(self._df))

        return self._df

    def get_graph(self, fetch=False, create_using=cugraph.Graph, ignore_weights=False):
        if self._graph is None:
            df = self.get_edgelist()
            self._graph = cugraph.Graph(directed=True)
            self._graph.from_cudf_edgelist(
                df, source="src", destination="dst", weight="weight")

    def get_path(self):
        return str(self)


rmat_dataset = pytest.param(RmatDataset(), marks=[pytest.rmat])

fixture_params = gen_fixture_params_product(
    (directed_datasets + undirected_datasets + [rmat_dataset], "ds"),
    (managed_memory, "mm"),
    (pool_allocator, "pa"))

# Record the current RMM settings so reinitialize() will be called only when a
# change is needed (RMM defaults both values to False). The --allow-rmm-reinit
# option is required to allow the RMM options to be set by the pytest user
# directly, in order to prevent reinitialize() from being called more than once
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
def edgelist(request):
    """
    Returns a new edgelist created from the dataset obj specified as part of
    the parameterization for this fixture.
    """
    # Since parameterized fixtures do not assign param names to param values,
    # manually call the helper to do so. Ensure the order of the name list
    # passed to it matches if there are >1 params.
    # If the request only contains n params, only the first n names are set.
    setFixtureParamNames(request, ["dataset", "managed_mem", "pool_allocator"])

    dataset = request.param[0]
    reinitRMM(request.param[1], request.param[2])

    if isinstance(dataset, RmatDataset):
        dataset.scale = request.config.getoption("--scale")
        dataset.edgefactor = request.config.getoption("--edgefactor")

    yield dataset.get_edgelist(fetch=True)
    dataset.unload()


@pytest.fixture(scope="module",
                params=fixture_params)
def graph(request):
    """
    Returns a new Graph with, adj list computed, created from the dataset obj
    specified as part of the parameterization for this fixture.
    """
    setFixtureParamNames(request, ["dataset", "managed_mem", "pool_allocator"])

    dataset = request.param[0]
    reinitRMM(request.param[1], request.param[2])

    if isinstance(dataset, RmatDataset):
        dataset.scale = request.config.getoption("--scale")
        dataset.edgefactor = request.config.getoption("--edgefactor")

    G = dataset.get_graph(fetch=True)
    G.view_adj_list()
    yield G
    dataset.unload()


@pytest.fixture(scope="module",
                params=fixture_params)
def graph_transposed(request):
    """
    Returns a new Graph with, transposed adj list computed, created from the
    dataset obj specified as part of the parameterization for this fixture.
    """
    setFixtureParamNames(request, ["dataset", "managed_mem", "pool_allocator"])

    dataset = request.param[0]
    reinitRMM(request.param[1], request.param[2])

    if isinstance(dataset, RmatDataset):
        dataset.scale = request.config.getoption("--scale")
        dataset.edgefactor = request.config.getoption("--edgefactor")

    G = dataset.get_graph(fetch=True)
    G.view_transposed_adj_list()
    yield G
    dataset.unload()




###############################################################################
# Benchmarks
@pytest.mark.ETL
def bench_create_graph(gpubenchmark, edgelist):
    gpubenchmark(cugraph.from_cudf_edgelist,
                 edgelist,
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
def bench_create_digraph(gpubenchmark, edgelist):
    gpubenchmark(cugraph.from_cudf_edgelist,
                 edgelist,
                 source="0", destination="1",
                 create_using=cugraph.Graph(directed=True),
                 renumber=False)


@pytest.mark.ETL
def bench_renumber(gpubenchmark, edgelist):
    gpubenchmark(NumberMap.renumber, edgelist, "0", "1")


def bench_pagerank(gpubenchmark, graph_transposed):
    gpubenchmark(cugraph.pagerank, graph_transposed)


def bench_bfs(gpubenchmark, graph):
    start = graph.edgelist.edgelist_df["src"][0]
    gpubenchmark(cugraph.bfs, graph, start)


def bench_force_atlas2(gpubenchmark, graph):
    gpubenchmark(cugraph.force_atlas2, graph,
                 max_iter=50)


def bench_sssp(gpubenchmark, graph):
    start = graph.edgelist.edgelist_df["src"][0]
    gpubenchmark(cugraph.sssp, graph, start)


def bench_jaccard(gpubenchmark, graph):
    gpubenchmark(cugraph.jaccard, graph)


@pytest.mark.skipif(
    is_device_version_less_than((7, 0)), reason="Not supported on Pascal")
def bench_louvain(gpubenchmark, graph):
    gpubenchmark(cugraph.louvain, graph)


def bench_weakly_connected_components(gpubenchmark,
                                      graph):
    if graph.is_directed():
        G = graph.to_undirected()
    else:
        G = graph
    gpubenchmark(cugraph.weakly_connected_components, G)


def bench_overlap(gpubenchmark, graph):
    gpubenchmark(cugraph.overlap, graph)


def bench_triangle_count(gpubenchmark, graph):
    gpubenchmark(cugraph.triangle_count, graph)


def bench_spectralBalancedCutClustering(gpubenchmark,
                                        graph):
    gpubenchmark(cugraph.spectralBalancedCutClustering,
                 graph, 2)


@pytest.mark.skip(reason="Need to guarantee graph has weights, "
                         "not doing that yet")
def bench_spectralModularityMaximizationClustering(
        gpubenchmark, graph):
    gpubenchmark(cugraph.spectralModularityMaximizationClustering,
                 graph, 2)


def bench_graph_degree(gpubenchmark, graph):
    gpubenchmark(graph.degree)


def bench_graph_degrees(gpubenchmark, graph):
    gpubenchmark(graph.degrees)


def bench_betweenness_centrality(gpubenchmark, graph):
    gpubenchmark(cugraph.betweenness_centrality,
                 graph, k=10, random_state=123)


def bench_edge_betweenness_centrality(gpubenchmark,
                                      graph):
    gpubenchmark(cugraph.edge_betweenness_centrality,
                 graph, k=10, seed=123)
