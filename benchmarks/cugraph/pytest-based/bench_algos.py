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
import numpy as np
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
import dask_cudf
from pylibcugraph.testing import gen_fixture_params_product

import cugraph
import cugraph.dask as dask_cugraph
from cugraph.structure.number_map import NumberMap
from cugraph.generators import rmat
from cugraph.testing import utils, mg_utils
from cugraph.utilities.utils import is_device_version_less_than

from cugraph_benchmarking.params import (
    directed_datasets,
    undirected_datasets,
    managed_memory,
    pool_allocator,
)

# duck-type compatible Dataset for RMAT data
class RmatDataset:
    def __init__(self, scale=4, edgefactor=2, mg=False):
        self._scale = scale
        self._edgefactor = edgefactor
        self._edgelist = None

        self.mg = mg

    def __str__(self):
        mg_str = "mg" if self.mg else "sg"
        return f"rmat_{mg_str}_{self._scale}_{self._edgefactor}"

    def get_edgelist(self, fetch=False):
        seed = 42
        if self._edgelist is None:
            self._edgelist = rmat(
                self._scale,
                (2**self._scale)*self._edgefactor,
                0.57,  # from Graph500
                0.19,  # from Graph500
                0.19,  # from Graph500
                seed or 42,
                clip_and_flip=False,
                scramble_vertex_ids=True,
                create_using=None,  # return edgelist instead of Graph instance
                mg=self.mg
            )
            rng = np.random.default_rng(seed)
            if self.mg:
                self._edgelist["weight"] = self._edgelist.map_partitions(
                    lambda df: rng.random(size=len(df)))
            else:
                self._edgelist["weight"] = rng.random(size=len(self._edgelist))

        return self._edgelist

    def get_graph(self,
                  fetch=False,
                  create_using=cugraph.Graph,
                  ignore_weights=False,
                  store_transposed=False):
        if isinstance(create_using, cugraph.Graph):
            # what about BFS if trnaposed is True
            attrs = {"directed": create_using.is_directed()}
            G = type(create_using)(**attrs)
        elif type(create_using) is type:
            G = create_using()

        edge_attr = None if ignore_weights else "weight"
        df = self.get_edgelist()
        if isinstance(df, dask_cudf.DataFrame):
            G.from_dask_cudf_edgelist(df,
                                      source="src",
                                      destination="dst",
                                      edge_attr=edge_attr,
                                      store_transposed=store_transposed)
        else:
            G.from_cudf_edgelist(df,
                                 source="src",
                                 destination="dst",
                                 edge_attr=edge_attr,
                                 store_transposed=store_transposed)
        return G

    def get_path(self):
        """
        (this is likely not needed for use with pytest-benchmark, just added for
        API completeness with Dataset.)
        """
        return str(self)

    def unload(self):
        self._edgelist = None


_rmat_scale = getattr(pytest, "_rmat_scale", 20)  # ~1M vertices
_rmat_edgefactor = getattr(pytest, "_rmat_edgefactor", 16)  # ~17M edges
rmat_sg_dataset = pytest.param(RmatDataset(scale=_rmat_scale,
                                           edgefactor=_rmat_edgefactor,
                                           mg=False),
                               marks=[pytest.mark.rmat_data,
                                      pytest.mark.sg,
                               ])
rmat_mg_dataset = pytest.param(RmatDataset(scale=_rmat_scale,
                                           edgefactor=_rmat_edgefactor,
                                           mg=True),
                               marks=[pytest.mark.rmat_data,
                                      pytest.mark.mg,
                               ])

rmm_fixture_params = gen_fixture_params_product(
    (managed_memory, "mm"),
    (pool_allocator, "pa"))
dataset_fixture_params = gen_fixture_params_product(
    (directed_datasets +
     undirected_datasets +
     [rmat_sg_dataset, rmat_mg_dataset], "ds"))

# Record the current RMM settings so reinitialize() will be called only when a
# change is needed (RMM defaults both values to False). The --allow-rmm-reinit
# option is required to allow the RMM options to be set by the pytest user
# directly, in order to prevent reinitialize() from being called more than once
# (see conftest.py for details).
# The defaults for managed_mem (False) and pool_alloc (True) are set in
# conftest.py
RMM_SETTINGS = {"managed_mem": False,
                "pool_alloc": False}

# FIXME: this only changes the RMM config in a SG environment. The dask config
# that applies to RMM in an MG environment is not changed by this!
def reinitRMM(managed_mem, pool_alloc):
    """
    Reinitializes RMM to the value of managed_mem and pool_alloc, but only if
    those values are different that the current configuration.
    """
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
                params=rmm_fixture_params)
def rmm_config(request):
    # Since parameterized fixtures do not assign param names to param values,
    # manually call the helper to do so. Ensure the order of the name list
    # passed to it matches if there are >1 params.
    # If the request only contains n params, only the first n names are set.
    setFixtureParamNames(request, ["managed_mem", "pool_allocator"])
    reinitRMM(request.param[0], request.param[1])


@pytest.fixture(scope="module",
                params=dataset_fixture_params)
def dataset(request, rmm_config):

    """
    Fixture which provides a Dataset instance, setting up a Dask cluster and
    client if necessary for MG, to tests and other fixtures. When all
    tests/fixtures are done with the Dataset, it has the Dask cluster and
    client torn down (if MG) and all data loaded is freed.
    """
    setFixtureParamNames(request, ["dataset"])
    dataset = request.param[0]
    client = cluster = None
    # For now, only RmatDataset instanaces support MG and have a "mg" attr.
    if hasattr(dataset, "mg") and dataset.mg:
        (client, cluster) = mg_utils.start_dask_client()

    yield dataset

    dataset.unload()
    if client is not None:
        mg_utils.stop_dask_client(client, cluster)


@pytest.fixture(scope="module")
def edgelist(request, dataset):
    df = dataset.get_edgelist()
    return df


@pytest.fixture(scope="module")
def graph(request, dataset):
    G = dataset.get_graph()
    return G


@pytest.fixture(scope="module")
def unweighted_graph(request, dataset):
    G = dataset.get_graph(ignore_weights=True)
    return G


@pytest.fixture(scope="module")
def directed_graph(request, dataset):
    G = dataset.get_graph(create_using=cugraph.Graph(directed=True))
    return G


@pytest.fixture(scope="module")
def transposed_graph(request, dataset):
    G = dataset.get_graph(store_transposed=True)
    return G


###############################################################################
def is_graph_distributed(graph):
    """
    Return True if graph is distributed (for use with cugraph.dask APIs)
    """
    return isinstance(graph.edgelist.edgelist_df, dask_cudf.DataFrame)


###############################################################################
# Benchmarks
def bench_create_graph(gpubenchmark, edgelist):
    gpubenchmark(cugraph.from_cudf_edgelist,
                 edgelist,
                 source="src", destination="dst",
                 create_using=cugraph.structure.graph_classes.Graph,
                 renumber=False)


# Creating directed Graphs on small datasets runs in micro-seconds, which
# results in thousands of rounds before the default threshold is met, so lower
# the max_time for this benchmark.
@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=10,
    max_time=0.005
)
def bench_create_digraph(gpubenchmark, edgelist):
    gpubenchmark(cugraph.from_cudf_edgelist,
                 edgelist,
                 source="src", destination="dst",
                 create_using=cugraph.Graph(directed=True),
                 renumber=False)


def bench_renumber(gpubenchmark, edgelist):
    gpubenchmark(NumberMap.renumber, edgelist, "src", "dst")


def bench_pagerank(gpubenchmark, transposed_graph):
    pagerank = dask_cugraph.pagerank if is_graph_distributed(transposed_graph) \
               else cugraph.pagerank
    gpubenchmark(pagerank, transposed_graph)


def bench_bfs(gpubenchmark, graph):
    bfs = dask_cugraph.bfs if is_graph_distributed(graph) else cugraph.bfs
    start = graph.edgelist.edgelist_df["src"][0]
    gpubenchmark(bfs, graph, start)


def bench_force_atlas2(gpubenchmark, graph):
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    gpubenchmark(cugraph.force_atlas2, graph, max_iter=50)


def bench_sssp(gpubenchmark, graph):
    sssp = dask_cugraph.sssp if is_graph_distributed(graph) else cugraph.sssp
    start = graph.edgelist.edgelist_df["src"][0]
    gpubenchmark(sssp, graph, start)


def bench_jaccard(gpubenchmark, unweighted_graph):
    G = unweighted_graph
    jaccard = dask_cugraph.jaccard if is_graph_distributed(G) else cugraph.jaccard
    gpubenchmark(jaccard, G)


@pytest.mark.skipif(
    is_device_version_less_than((7, 0)), reason="Not supported on Pascal")
def bench_louvain(gpubenchmark, graph):
    louvain = dask_cugraph.louvain if is_graph_distributed(graph) else cugraph.louvain
    gpubenchmark(louvain, graph)


def bench_weakly_connected_components(gpubenchmark, graph):
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    if graph.is_directed():
        G = graph.to_undirected()
    else:
        G = graph
    gpubenchmark(cugraph.weakly_connected_components, G)


def bench_overlap(gpubenchmark, unweighted_graph):
    G = unweighted_graph
    overlap = dask_cugraph.overlap if is_graph_distributed(G) else cugraph.overlap
    gpubenchmark(overlap, G)


def bench_triangle_count(gpubenchmark, graph):
    tc = dask_cugraph.triangle_count if is_graph_distributed(graph) \
         else cugraph.triangle_count
    gpubenchmark(tc, graph)


def bench_spectralBalancedCutClustering(gpubenchmark, graph):
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    gpubenchmark(cugraph.spectralBalancedCutClustering, graph, 2)


@pytest.mark.skip(reason="Need to guarantee graph has weights, "
                         "not doing that yet")
def bench_spectralModularityMaximizationClustering(gpubenchmark, graph):
    smmc = dask_cugraph.spectralModularityMaximizationClustering \
           if is_graph_distributed(graph) \
           else cugraph.spectralModularityMaximizationClustering
    gpubenchmark(smmc, graph, 2)


def bench_graph_degree(gpubenchmark, graph):
    gpubenchmark(graph.degree)


def bench_graph_degrees(gpubenchmark, graph):
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    gpubenchmark(graph.degrees)


def bench_betweenness_centrality(gpubenchmark, graph):
    bc = dask_cugraph.betweenness_centrality if is_graph_distributed(graph) \
         else cugraph.betweenness_centrality
    gpubenchmark(bc, graph, k=10, random_state=123)


def bench_edge_betweenness_centrality(gpubenchmark, graph):
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    gpubenchmark(cugraph.edge_betweenness_centrality, graph, k=10, seed=123)


def bench_uniform_neighbor_sample(gpubenchmark, graph):
    uns = dask_cugraph.uniform_neighbor_sample if is_graph_distributed(graph) \
         else cugraph.uniform_neighbor_sample

    seed = 42
    # FIXME: may need to provide number_of_vertices separately
    num_verts_in_graph = graph.number_of_vertices()
    len_start_list = max(int(num_verts_in_graph * 0.01), 2)
    srcs = graph.edgelist.edgelist_df["src"]
    frac = len_start_list / num_verts_in_graph

    start_list = srcs.sample(frac=frac, random_state=seed)
    # Attempt to automatically handle a dask Series
    if hasattr(start_list, "compute"):
        start_list = start_list.compute()

    fanout_vals = [5, 5, 5]
    gpubenchmark(uns, graph, start_list=start_list, fanout_vals=fanout_vals)


def bench_egonet(gpubenchmark, graph):
    egonet = dask_cugraph.ego_graph if is_graph_distributed(graph) \
             else cugraph.ego_graph
    n = 1
    radius = 2
    gpubenchmark(egonet, graph, n, radius=radius)
