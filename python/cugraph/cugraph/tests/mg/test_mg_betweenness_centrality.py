# Copyright (c) 2022, NVIDIA CORPORATION.
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

import cugraph.dask as dcg
import gc
import pytest
import cugraph
import dask_cudf
import random

# from cugraph.dask.common.mg_utils import is_single_gpu
from cugraph.testing import utils


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


IS_DIRECTED = [True, False]


# =============================================================================
# Pytest fixtures
# =============================================================================

datasets = utils.DATASETS_UNDIRECTED + [
    utils.RAPIDS_DATASET_ROOT_DIR_PATH / "email-Eu-core.csv"
]

fixture_params = utils.genFixtureParamsProduct(
    (datasets, "graph_file"),
    ([False, True], "has_vertex_list"),
    ([True, False], "normalized"),
    ([True, False], "endpoints"),
    (IS_DIRECTED, "directed"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(
        zip(
            ("graph_file", "has_vertex_list", "normalized", "endpoints", "directed"),
            request.param,
        )
    )

    return parameters


@pytest.fixture(scope="module")
def input_expected_output(input_combo):
    """
    This fixture returns the inputs and expected results from the
    betweenness_centrality algo based on cuGraph betweenness_centrality) which can
    be used for validation.
    """

    input_data_path = input_combo["graph_file"]
    directed = input_combo["directed"]
    normalized = input_combo["normalized"]
    endpoints = input_combo["endpoints"]
    vertex_list = None

    input_combo["vertex_list"] = vertex_list
    G = utils.generate_cugraph_graph_from_file(input_data_path, directed=directed)

    if input_combo["has_vertex_list"]:
        # Sample vertices from the graph
        k = random.randint(1, 4)
        vertex_list = G.nodes().compute().sample(k).reset_index(drop=True)

    sg_cugraph_bc = cugraph.betweenness_centrality(
        G, vertex_list=vertex_list, normalized=normalized, endpoints=endpoints
    )
    # Save the results back to the input_combo dictionary to prevent redundant
    # cuGraph runs. Other tests using the input_combo fixture will look for
    # them, and if not present they will have to re-run the same cuGraph call.
    sg_cugraph_bc = sg_cugraph_bc.sort_values("vertex").reset_index(drop=True)

    input_combo["sg_cugraph_results"] = sg_cugraph_bc
    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    dg = cugraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(
        ddf,
        source="src",
        destination="dst",
        edge_attr="value",
        renumber=True,
        legacy_renum_only=True,
        store_transposed=True,
    )

    input_combo["MGGraph"] = dg

    return input_combo


# =============================================================================
# Tests
# =============================================================================


# @pytest.mark.skipif(
#    is_single_gpu(), reason="skipping MG testing on Single GPU system"
# )
def test_dask_betweenness_centrality(dask_client, benchmark, input_expected_output):

    dg = input_expected_output["MGGraph"]
    vertex_list = input_expected_output["vertex_list"]
    endpoints = input_expected_output["endpoints"]
    normalized = input_expected_output["normalized"]

    result_betweenness_centrality = benchmark(
        dcg.betweenness_centrality,
        dg,
        vertex_list=vertex_list,
        normalized=normalized,
        endpoints=normalized,
    )

    result_betweenness_centrality = (
        result_betweenness_centrality.compute()
        .sort_values("vertex")
        .reset_index(drop=True)
        .rename(
            columns={"hubs": "mg_cugraph_hubs", "authorities": "mg_cugraph_authorities"}
        )
    )

    expected_output = (
        input_expected_output["sg_cugraph_results"]
        .sort_values("vertex")
        .reset_index(drop=True)
    )

    # Update the dask cugraph betweenness_centrality results with sg cugraph results for easy
    # comparison using cuDF DataFrame methods.
    result_betweenness_centrality["sg_cugraph_hubs"] = expected_output["hubs"]
    result_betweenness_centrality["sg_cugraph_authorities"] = expected_output[
        "authorities"
    ]

    hubs_diffs1 = result_betweenness_centrality.query(
        "mg_cugraph_hubs - sg_cugraph_hubs > 0.00001"
    )
    hubs_diffs2 = result_betweenness_centrality.query(
        "mg_cugraph_hubs - sg_cugraph_hubs < -0.00001"
    )
    authorities_diffs1 = result_betweenness_centrality.query(
        "mg_cugraph_authorities - sg_cugraph_authorities > 0.0001"
    )
    authorities_diffs2 = result_betweenness_centrality.query(
        "mg_cugraph_authorities - sg_cugraph_authorities < -0.0001"
    )

    assert len(hubs_diffs1) == 0
    assert len(hubs_diffs2) == 0
    assert len(authorities_diffs1) == 0
    assert len(authorities_diffs2) == 0


def test_dask_hots_transposed_True(dask_client):
    input_data_path = (utils.RAPIDS_DATASET_ROOT_DIR_PATH / "karate.csv").as_posix()

    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    dg = cugraph.Graph(directed=True)
    dg.from_dask_cudf_edgelist(
        ddf, "src", "dst", legacy_renum_only=True, store_transposed=True
    )

    warning_msg = (
        "betweenness_centrality expects the 'store_transposed' "
        "flag to be set to 'False' for optimal performance during "
        "the graph creation"
    )

    with pytest.warns(UserWarning, match=warning_msg):
        dcg.betweenness_centrality(dg)
