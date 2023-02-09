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

import gc

import pytest
import networkx as nx
import pandas as pd
import cudf
from pylibcugraph.testing.utils import gen_fixture_params_product

import cugraph
from cugraph.testing import utils
from cugraph.experimental.datasets import DATASETS_UNDIRECTED, email_Eu_core, karate


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Pytest fixtures
# =============================================================================
datasets = DATASETS_UNDIRECTED + [email_Eu_core]
fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    ([50], "max_iter"),
    ([1.0e-6], "tol"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    return dict(zip(("graph_file", "max_iter", "tol"), request.param))


@pytest.fixture(scope="module")
def input_expected_output(input_combo):
    """
    This fixture returns a dictionary containing all input params required to
    run a HITS algo and the corresponding expected result (based on NetworkX
    HITS) which can be used for validation.
    """
    # Only run Nx to compute the expected result if it is not already present
    # in the dictionary. This allows separate Nx-only tests that may have run
    # previously on the same input_combo to save their results for re-use
    # elsewhere.
    if "nxResults" not in input_combo:
        dataset_path = input_combo["graph_file"].get_path()
        Gnx = utils.generate_nx_graph_from_file(dataset_path, directed=True)
        nxResults = nx.hits(
            Gnx, input_combo["max_iter"], input_combo["tol"], normalized=True
        )
        input_combo["nxResults"] = nxResults
    return input_combo


# =============================================================================
# Tests
# =============================================================================
def test_nx_hits(benchmark, input_combo):
    """
    Simply run NetworkX HITS on the same set of input combinations used for the
    cuGraph HITS tests.
    This is only in place for generating comparison performance numbers.
    """
    dataset_path = input_combo["graph_file"].get_path()
    Gnx = utils.generate_nx_graph_from_file(dataset_path, directed=True)
    nxResults = benchmark(
        nx.hits, Gnx, input_combo["max_iter"], input_combo["tol"], normalized=True
    )
    # Save the results back to the input_combo dictionary to prevent redundant
    # Nx runs. Other tests using the input_combo fixture will look for them,
    # and if not present they will have to re-run the same Nx call.
    input_combo["nxResults"] = nxResults


def test_hits(benchmark, input_expected_output):
    graph_file = input_expected_output["graph_file"]

    G = graph_file.get_graph(create_using=cugraph.Graph(directed=True))
    cugraph_hits = benchmark(
        cugraph.hits, G, input_expected_output["max_iter"], input_expected_output["tol"]
    )
    cugraph_hits = cugraph_hits.sort_values("vertex").reset_index(drop=True)

    (nx_hubs, nx_authorities) = input_expected_output["nxResults"]

    # Update the cugraph HITS results with Nx results for easy comparison using
    # cuDF DataFrame methods.
    pdf = pd.DataFrame.from_dict(nx_hubs, orient="index").sort_index()
    cugraph_hits["nx_hubs"] = cudf.Series.from_pandas(pdf[0])
    pdf = pd.DataFrame.from_dict(nx_authorities, orient="index").sort_index()
    cugraph_hits["nx_authorities"] = cudf.Series.from_pandas(pdf[0])
    hubs_diffs1 = cugraph_hits.query("hubs - nx_hubs > 0.00001")
    hubs_diffs2 = cugraph_hits.query("hubs - nx_hubs < -0.00001")
    authorities_diffs1 = cugraph_hits.query("authorities - nx_authorities > 0.0001")
    authorities_diffs2 = cugraph_hits.query("authorities - nx_authorities < -0.0001")

    assert len(hubs_diffs1) == 0
    assert len(hubs_diffs2) == 0
    assert len(authorities_diffs1) == 0
    assert len(authorities_diffs2) == 0


def test_hits_transposed_false():

    G = karate.get_graph(create_using=cugraph.Graph(directed=True))
    warning_msg = (
        "Pagerank expects the 'store_transposed' "
        "flag to be set to 'True' for optimal performance during "
        "the graph creation"
    )

    with pytest.warns(UserWarning, match=warning_msg):
        cugraph.pagerank(G)
