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
import cudf
from pylibcugraph.testing.utils import gen_fixture_params_product
import networkx as nx

import cugraph
from cugraph.testing import utils
from cugraph.experimental.datasets import DATASETS_UNDIRECTED


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Pytest fixtures
# =============================================================================
datasets = DATASETS_UNDIRECTED
degree_type = ["incoming", "outgoing"]

fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    (degree_type, "degree_type"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    This fixture returns a dictionary containing all input params required to
    run a Core number algo
    """
    parameters = dict(zip(("graph_file", "degree_type"), request.param))

    graph_file = parameters["graph_file"]
    G = graph_file.get_graph()
    input_data_path = graph_file.get_path()

    Gnx = utils.generate_nx_graph_from_file(
        input_data_path, directed=False, edgevals=True
    )

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

    dic_results = nx.core_number(Gnx)
    nx_core_number_results["vertex"] = dic_results.keys()
    nx_core_number_results["core_number"] = dic_results.values()
    nx_core_number_results = nx_core_number_results.sort_values("vertex").reset_index(
        drop=True
    )

    core_number_results = (
        cugraph.core_number(G, degree_type)
        .sort_values("vertex")
        .reset_index(drop=True)
        .rename(columns={"core_number": "cugraph_core_number"})
    )

    # Compare the nx core number results with cugraph
    core_number_results["nx_core_number"] = nx_core_number_results["core_number"]

    counts_diff = core_number_results.query("nx_core_number != cugraph_core_number")
    assert len(counts_diff) == 0


def test_core_number_invalid_input(input_combo):
    input_data_path = (
        utils.RAPIDS_DATASET_ROOT_DIR_PATH / "karate-asymmetric.csv"
    ).as_posix()
    M = utils.read_csv_for_nx(input_data_path)
    G = cugraph.Graph(directed=True)
    cu_M = cudf.DataFrame()
    cu_M["src"] = cudf.Series(M["0"])
    cu_M["dst"] = cudf.Series(M["1"])

    cu_M["weights"] = cudf.Series(M["weight"])
    G.from_cudf_edgelist(cu_M, source="src", destination="dst", edge_attr="weights")

    with pytest.raises(ValueError):
        cugraph.core_number(G)

    invalid_degree_type = "invalid"
    G = input_combo["G"]
    with pytest.raises(ValueError):
        cugraph.core_number(G, invalid_degree_type)
