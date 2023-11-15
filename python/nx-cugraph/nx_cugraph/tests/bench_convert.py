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
import numpy as np
import pytest

import nx_cugraph as nxcg

try:
    import cugraph
except ModuleNotFoundError:
    cugraph = None
try:
    import scipy
except ModuleNotFoundError:
    scipy = None

# If the rapids-pytest-benchmark plugin is installed, the "gpubenchmark"
# fixture will be available automatically. Check that this fixture is available
# by trying to import rapids_pytest_benchmark, and if that fails, set
# "gpubenchmark" to the standard "benchmark" fixture provided by
# pytest-benchmark.
try:
    import rapids_pytest_benchmark  # noqa: F401
except ModuleNotFoundError:
    import pytest_benchmark

    gpubenchmark = pytest_benchmark.plugin.benchmark

CREATE_USING = [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]


def _bench_helper(gpubenchmark, N, attr_kind, create_using, method):
    G = method(N, create_using=create_using)
    if attr_kind:
        skip = True
        for *_ids, edgedict in G.edges(data=True):
            skip = not skip
            if skip and attr_kind not in {"full", "required", "required_dtype"}:
                continue
            edgedict["x"] = random.randint(0, 100000)
        if attr_kind == "preserve":
            gpubenchmark(nxcg.from_networkx, G, preserve_edge_attrs=True)
        elif attr_kind == "half_missing":
            gpubenchmark(nxcg.from_networkx, G, edge_attrs={"x": None})
        elif attr_kind == "required":
            gpubenchmark(nxcg.from_networkx, G, edge_attrs={"x": ...})
        elif attr_kind == "required_dtype":
            gpubenchmark(
                nxcg.from_networkx,
                G,
                edge_attrs={"x": ...},
                edge_dtypes={"x": np.int32},
            )
        else:  # full, half_default
            gpubenchmark(nxcg.from_networkx, G, edge_attrs={"x": 0})
    else:
        gpubenchmark(nxcg.from_networkx, G)


def _bench_helper_cugraph(
    gpubenchmark, N, attr_kind, create_using, method, do_renumber
):
    G = method(N, create_using=create_using)
    if attr_kind:
        for *_ids, edgedict in G.edges(data=True):
            edgedict["x"] = random.randint(0, 100000)
        gpubenchmark(cugraph.utilities.convert_from_nx, G, "x", do_renumber=do_renumber)
    else:
        gpubenchmark(cugraph.utilities.convert_from_nx, G, do_renumber=do_renumber)


def _bench_helper_scipy(gpubenchmark, N, attr_kind, create_using, method, fmt):
    G = method(N, create_using=create_using)
    if attr_kind:
        for *_ids, edgedict in G.edges(data=True):
            edgedict["x"] = random.randint(0, 100000)
        gpubenchmark(nx.to_scipy_sparse_array, G, weight="x", format=fmt)
    else:
        gpubenchmark(nx.to_scipy_sparse_array, G, weight=None, format=fmt)


@pytest.mark.parametrize("N", [1, 10**6])
@pytest.mark.parametrize(
    "attr_kind",
    [
        "required_dtype",
        "required",
        "full",
        "half_missing",
        "half_default",
        "preserve",
        None,
    ],
)
@pytest.mark.parametrize("create_using", CREATE_USING)
def bench_cycle_graph(gpubenchmark, N, attr_kind, create_using):
    _bench_helper(gpubenchmark, N, attr_kind, create_using, nx.cycle_graph)


@pytest.mark.skipif("not cugraph")
@pytest.mark.parametrize("N", [1, 10**6])
@pytest.mark.parametrize("attr_kind", ["full", None])
@pytest.mark.parametrize("create_using", CREATE_USING)
@pytest.mark.parametrize("do_renumber", [True, False])
def bench_cycle_graph_cugraph(gpubenchmark, N, attr_kind, create_using, do_renumber):
    if N == 1 and not do_renumber:
        do_renumber = True
    _bench_helper_cugraph(
        gpubenchmark, N, attr_kind, create_using, nx.cycle_graph, do_renumber
    )


@pytest.mark.skipif("not scipy")
@pytest.mark.parametrize("N", [1, 10**6])
@pytest.mark.parametrize("attr_kind", ["full", None])
@pytest.mark.parametrize("create_using", CREATE_USING)
@pytest.mark.parametrize("fmt", ["coo", "csr"])
def bench_cycle_graph_scipy(gpubenchmark, N, attr_kind, create_using, fmt):
    _bench_helper_scipy(gpubenchmark, N, attr_kind, create_using, nx.cycle_graph, fmt)


@pytest.mark.parametrize("N", [1, 1500])
@pytest.mark.parametrize(
    "attr_kind",
    [
        "required_dtype",
        "required",
        "full",
        "half_missing",
        "half_default",
        "preserve",
        None,
    ],
)
@pytest.mark.parametrize("create_using", CREATE_USING)
def bench_complete_graph_edgedata(gpubenchmark, N, attr_kind, create_using):
    _bench_helper(gpubenchmark, N, attr_kind, create_using, nx.complete_graph)


@pytest.mark.parametrize("N", [3000])
@pytest.mark.parametrize("attr_kind", [None])
@pytest.mark.parametrize("create_using", CREATE_USING)
def bench_complete_graph_noedgedata(gpubenchmark, N, attr_kind, create_using):
    _bench_helper(gpubenchmark, N, attr_kind, create_using, nx.complete_graph)


@pytest.mark.skipif("not cugraph")
@pytest.mark.parametrize("N", [1, 1500])
@pytest.mark.parametrize("attr_kind", ["full", None])
@pytest.mark.parametrize("create_using", CREATE_USING)
@pytest.mark.parametrize("do_renumber", [True, False])
def bench_complete_graph_cugraph(gpubenchmark, N, attr_kind, create_using, do_renumber):
    if N == 1 and not do_renumber:
        do_renumber = True
    _bench_helper_cugraph(
        gpubenchmark, N, attr_kind, create_using, nx.complete_graph, do_renumber
    )


@pytest.mark.skipif("not scipy")
@pytest.mark.parametrize("N", [1, 1500])
@pytest.mark.parametrize("attr_kind", ["full", None])
@pytest.mark.parametrize("create_using", CREATE_USING)
@pytest.mark.parametrize("fmt", ["coo", "csr"])
def bench_complete_graph_scipy(gpubenchmark, N, attr_kind, create_using, fmt):
    _bench_helper_scipy(
        gpubenchmark, N, attr_kind, create_using, nx.complete_graph, fmt
    )
