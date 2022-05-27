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

import pytest
import random

import cudf
import cugraph
from cugraph.testing import utils
from cugraph.experimental import triangle_count as experimental_triangles


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
datasets = utils.DATASETS_UNDIRECTED
fixture_params = utils.genFixtureParamsProduct((datasets, "graph_file"),
                                               ([True, False], "edgevals"),
                                               )


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    return dict(zip(("graph_file", "edgevals", "start_list"), request.param))


@pytest.fixture(scope="module")
def input_expected_output(input_combo):
    """
    This fixture returns a dictionary containing all input params required to
    run a Triangle Count algo
    """
    input_data_path = input_combo["graph_file"]
    edgevals = input_combo["edgevals"]

    G = utils.generate_cugraph_graph_from_file(
        input_data_path, directed=False, edgevals=edgevals)

    Gnx = utils.generate_nx_graph_from_file(
        input_data_path, directed=False, edgevals=edgevals)

    input_combo["G"] = G
    input_combo["Gnx"] = Gnx

    return input_combo


# =============================================================================
# Tests
# =============================================================================
def test_triangles_no_start(input_expected_output):

    G = input_expected_output["G"]
    Gnx = input_expected_output["Gnx"]
    nx_triangle_results = cudf.DataFrame()

    cugraph_legacy_triangle_results = cugraph.triangles(G)

    dic_results = nx.triangles(Gnx)
    nx_triangle_results["vertex"] = dic_results.keys()
    nx_triangle_results["counts"] = dic_results.values()
    nx_triangle_results = nx_triangle_results.sort_values(
        "vertex").reset_index(drop=True)

    assert cugraph_legacy_triangle_results == \
        nx_triangle_results["counts"].sum()

    if input_expected_output["edgevals"]:
        triangle_results = experimental_triangles(G).sort_values(
            "vertex").reset_index(drop=True).rename(columns={
                "counts": "exp_cugraph_counts"})
        cugraph_exp_triangle_results = \
            triangle_results["exp_cugraph_counts"].sum()
        # Compare the total number of triangles with the experimental
        # implementation
        assert cugraph_exp_triangle_results == nx_triangle_results
        # Compare the number of triangles per vertex with the
        # experimental implementation
        triangle_results["nx_counts"] = nx_triangle_results["counts"]
        counts_diff = triangle_results.query(
            'nx_counts != exp_cugraph_counts')
        assert len(counts_diff) == 0


def test_triangles_with_start(input_expected_output):
    if input_expected_output["edgevals"]:
        G = input_expected_output["G"]
        Gnx = input_expected_output["Gnx"]
        nx_triangle_results = cudf.DataFrame()

        # sample k nodes from the nx graph
        k = random.randint(1, 10)
        start_list = random.sample(list(Gnx.nodes()), k)

        dic_results = nx.triangles(Gnx, start_list)
        nx_triangle_results["vertex"] = dic_results.keys()
        nx_triangle_results["counts"] = dic_results.values()
        nx_triangle_results = nx_triangle_results.sort_values(
            "vertex").reset_index(drop=True)

        start_list = cudf.Series(start_list, dtype="int32")
        triangle_results = experimental_triangles(
            G, start_list).sort_values("vertex").reset_index(
                drop=True).rename(columns={"counts": "exp_cugraph_counts"})

        triangle_results["nx_counts"] = nx_triangle_results["counts"]
        counts_diff = triangle_results.query(
            'nx_counts != exp_cugraph_counts')
        assert len(counts_diff) == 0


def test_triangles_directed_graph():
    input_data_path = (utils.RAPIDS_DATASET_ROOT_DIR_PATH /
                       "karate-asymmetric.csv").as_posix()
    M = utils.read_csv_for_nx(input_data_path)
    G = cugraph.Graph(directed=True)
    cu_M = cudf.DataFrame()
    cu_M["src"] = cudf.Series(M["0"])
    cu_M["dst"] = cudf.Series(M["1"])

    cu_M["weights"] = cudf.Series(M["weight"])
    G.from_cudf_edgelist(
        cu_M, source="src", destination="dst", edge_attr="weights"
    )

    with pytest.raises(ValueError):
        cugraph.triangles(G)

    with pytest.raises(ValueError):
        experimental_triangles(G)
