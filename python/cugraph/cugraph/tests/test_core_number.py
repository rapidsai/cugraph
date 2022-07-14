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
from cugraph.experimental import core_number as experimental_core_number


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
# FIXME: The `start_list` parameter is not supported yet therefore it has been
# disabled in these tests. Enable it once it is supported
fixture_params = utils.genFixtureParamsProduct((datasets, "graph_file"),
                                               ([0,1], "degree_type"),
                                               )


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    This fixture returns a dictionary containing all input params required to
    run a Triangle Count algo
    """
    parameters = dict(
        zip(("graph_file", "degree_type"), request.param))

    input_data_path = parameters["graph_file"]

    G = utils.generate_cugraph_graph_from_file(
        input_data_path, directed=False, edgevals=True)

    Gnx = utils.generate_nx_graph_from_file(
        input_data_path, directed=False, edgevals=True)

    parameters["G"] = G
    parameters["Gnx"] = Gnx

    return parameters


# =============================================================================
# Tests
# =============================================================================
def test_core_number(input_combo):
    G = input_combo["G"]
    Gnx = input_combo["Gnx"]
    degree_type = input_combo["degree_type"]
    nx_core_number_results = cudf.DataFrame()

    cugraph_legacy_core_number_results = cugraph.core_number(G).sort_values(
        "vertex").reset_index(drop=True)

    dic_results = nx.core_number(Gnx)
    nx_core_number_results["vertex"] = dic_results.keys()
    nx_core_number_results["core_number"] = dic_results.values()
    nx_core_number_results = nx_core_number_results.sort_values(
        "vertex").reset_index(drop=True)

    assert cugraph_legacy_core_number_results == \
        nx_core_number_results["core_number"].sum()

    core_number_results = experimental_core_number(G, degree_type).sort_values(
        "vertex").reset_index(drop=True).rename(columns={
            "core_number": "exp_cugraph_core_number"})

    cugraph_exp_core_number_results = \
        core_number_results["exp_cugraph_core_number"].sum()
    # Compare the total number of triangles with the experimental
    # implementation
    assert cugraph_exp_core_number_results == nx_core_number_results
    # Compare the number of triangles per vertex with the
    # experimental implementation
    core_number_results["nx_core_number"] = nx_core_number_results["core_number"]
    core_number_results["legacy_cugraph_core_number"] = cugraph_legacy_core_number_results["core_number"]
    print("core number results is \n", core_number_results)
    counts_diff = core_number_results.query(
        'nx_core_number != exp_cugraph_core_number or exp_cugraph_core_number != legacy_cugraph_core_number')
    assert len(counts_diff) == 0
    
    #print("core number results is \n", core_number_results)


"""
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
"""