# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

import gc
import random

import pytest
import cudf
from pylibcugraph.testing.utils import gen_fixture_params_product

import cugraph
from cugraph.testing import utils
from cugraph.experimental.datasets import DATASETS_UNDIRECTED, karate_asymmetric


# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Pytest fixtures
# =============================================================================
datasets = DATASETS_UNDIRECTED
fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    ([True, False], "edgevals"),
    ([True, False], "start_list"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    This fixture returns a dictionary containing all input params required to
    run a Triangle Count algo
    """
    parameters = dict(zip(("graph_file", "edgevals", "start_list"), request.param))

    graph_file = parameters["graph_file"]
    input_data_path = graph_file.get_path()
    edgevals = parameters["edgevals"]

    G = graph_file.get_graph(ignore_weights=not edgevals)

    Gnx = utils.generate_nx_graph_from_file(
        input_data_path, directed=False, edgevals=edgevals
    )

    parameters["G"] = G
    parameters["Gnx"] = Gnx

    return parameters


# =============================================================================
# Tests
# =============================================================================
def test_triangles(input_combo):
    G = input_combo["G"]
    Gnx = input_combo["Gnx"]
    nx_triangle_results = cudf.DataFrame()

    if input_combo["start_list"]:
        # sample k nodes from the nx graph
        k = random.randint(1, 10)
        start_list = random.sample(list(Gnx.nodes()), k)
    else:
        start_list = None

    cugraph_triangle_results = cugraph.triangle_count(G, start_list)

    triangle_results = (
        cugraph_triangle_results.sort_values("vertex")
        .reset_index(drop=True)
        .rename(columns={"counts": "cugraph_counts"})
    )

    dic_results = nx.triangles(Gnx, start_list)
    nx_triangle_results["vertex"] = dic_results.keys()
    nx_triangle_results["counts"] = dic_results.values()
    nx_triangle_results = nx_triangle_results.sort_values("vertex").reset_index(
        drop=True
    )

    triangle_results["nx_counts"] = nx_triangle_results["counts"]
    counts_diff = triangle_results.query("nx_counts != cugraph_counts")
    assert len(counts_diff) == 0


def test_triangles_int64(input_combo):
    Gnx = input_combo["Gnx"]
    count_legacy_32 = cugraph.triangle_count(Gnx)

    graph_file = input_combo["graph_file"]
    G = graph_file.get_graph()
    G.edgelist.edgelist_df = G.edgelist.edgelist_df.astype(
        {"src": "int64", "dst": "int64"}
    )

    count_exp_64 = (
        cugraph.triangle_count(G)
        .sort_values("vertex")
        .reset_index(drop=True)
        .rename(columns={"counts": "exp_cugraph_counts"})
    )
    cugraph_exp_triangle_results = count_exp_64["exp_cugraph_counts"].sum()
    assert G.edgelist.edgelist_df["src"].dtype == "int64"
    assert G.edgelist.edgelist_df["dst"].dtype == "int64"
    assert cugraph_exp_triangle_results == count_legacy_32


def test_triangles_no_weights(input_combo):
    G_weighted = input_combo["Gnx"]
    count_legacy = (
        cugraph.triangle_count(G_weighted)
        .sort_values("vertex")
        .reset_index(drop=True)
        .rename(columns={"counts": "exp_cugraph_counts"})
    )

    graph_file = input_combo["graph_file"]
    G = graph_file.get_graph(ignore_weights=True)

    assert G.is_weighted() is False
    triangle_count = (
        cugraph.triangle_count(G)
        .sort_values("vertex")
        .reset_index(drop=True)
        .rename(columns={"counts": "exp_cugraph_counts"})
    )
    cugraph_exp_triangle_results = triangle_count["exp_cugraph_counts"].sum()
    assert cugraph_exp_triangle_results == count_legacy


def test_triangles_directed_graph():
    input_data_path = karate_asymmetric.get_path()
    M = utils.read_csv_for_nx(input_data_path)
    G = cugraph.Graph(directed=True)
    cu_M = cudf.DataFrame()
    cu_M["src"] = cudf.Series(M["0"])
    cu_M["dst"] = cudf.Series(M["1"])

    cu_M["weights"] = cudf.Series(M["weight"])
    G.from_cudf_edgelist(cu_M, source="src", destination="dst", edge_attr="weights")

    with pytest.raises(ValueError):
        cugraph.triangle_count(G)


# FIXME: Remove this test once experimental.triangle count is removed
def test_experimental_triangle_count(input_combo):
    G = input_combo["G"]
    with pytest.warns(Warning):
        cugraph.experimental.triangle_count(G)
