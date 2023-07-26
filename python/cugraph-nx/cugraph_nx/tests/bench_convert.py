# Copyright (c) 2023, NVIDIA CORPORATION.
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
import random

import networkx as nx
import pytest

import cugraph_nx as cnx

try:
    import cugraph
except ImportError:
    cugraph = None

# If the rapids-pytest-benchmark plugin is installed, the "gpubenchmark"
# fixture will be available automatically. Check that this fixture is available
# by trying to import rapids_pytest_benchmark, and if that fails, set
# "gpubenchmark" to the standard "benchmark" fixture provided by
# pytest-benchmark.
try:
    import rapids_pytest_benchmark  # noqa: F401
except ImportError:
    import pytest_benchmark

    gpubenchmark = pytest_benchmark.plugin.benchmark


def _bench_helper(gpubenchmark, N, attr_kind, create_using, method):
    G = method(N, create_using=create_using)
    if attr_kind:
        skip = True
        for *_ids, edgedict in G.edges(data=True):
            skip = not skip
            if skip and attr_kind is not True:
                continue
            edgedict["x"] = random.randint(0, 100000)
        if attr_kind == "preserve":
            gpubenchmark(cnx.from_networkx, G, preserve_edge_attrs=True)
        elif attr_kind == "half_missing":
            gpubenchmark(cnx.from_networkx, G, edge_attrs={"x": None})
        else:
            gpubenchmark(cnx.from_networkx, G, edge_attrs={"x": 0})
    else:
        gpubenchmark(cnx.from_networkx, G)


def _bench_helper_cugraph(gpubenchmark, N, attr_kind, create_using, method):
    G = method(N, create_using=create_using)
    if attr_kind:
        for *_ids, edgedict in G.edges(data=True):
            edgedict["x"] = random.randint(0, 100000)
        gpubenchmark(cugraph.utilities.convert_from_nx, G, "x")
    else:
        gpubenchmark(cugraph.utilities.convert_from_nx, G)


@pytest.mark.parametrize("N", [1, 10**2, 10**4])
@pytest.mark.parametrize(
    "attr_kind", ["full", "half_missing", "half_default", "preserve", None]
)
@pytest.mark.parametrize("create_using", [nx.Graph, nx.DiGraph])
def bench_cycle_graph(gpubenchmark, N, attr_kind, create_using):
    _bench_helper(gpubenchmark, N, attr_kind, create_using, nx.cycle_graph)


@pytest.mark.skipif("not cugraph")
@pytest.mark.parametrize("N", [1, 10**2, 10**4])
@pytest.mark.parametrize("attr_kind", ["full", None])
@pytest.mark.parametrize("create_using", [nx.Graph, nx.DiGraph])
def bench_cycle_graph_cugraph(gpubenchmark, N, attr_kind, create_using):
    _bench_helper_cugraph(gpubenchmark, N, attr_kind, create_using, nx.cycle_graph)


@pytest.mark.parametrize("N", [1, 30, 1000])
@pytest.mark.parametrize(
    "attr_kind", ["full", "half_missing", "half_default", "preserve", None]
)
@pytest.mark.parametrize("create_using", [nx.Graph, nx.DiGraph])
def bench_complete_graph(gpubenchmark, N, attr_kind, create_using):
    _bench_helper(gpubenchmark, N, attr_kind, create_using, nx.complete_graph)


@pytest.mark.parametrize("N", [1, 30, 1000])
@pytest.mark.parametrize("attr_kind", ["full", None])
@pytest.mark.parametrize("create_using", [nx.Graph, nx.DiGraph])
def bench_complete_graph_cugraph(gpubenchmark, N, attr_kind, create_using):
    _bench_helper_cugraph(gpubenchmark, N, attr_kind, create_using, nx.complete_graph)
