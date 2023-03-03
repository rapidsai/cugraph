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

import gc
import random
import pytest

import dask_cudf
from pylibcugraph.testing import gen_fixture_params_product

import cugraph.dask as dcg
import cugraph
from cugraph.testing import utils


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


IS_DIRECTED = [False]
HAS_VERTEX_PAIR = [True, False]


# =============================================================================
# Pytest fixtures
# =============================================================================

datasets = utils.DATASETS_UNDIRECTED + [
    utils.RAPIDS_DATASET_ROOT_DIR_PATH / "email-Eu-core.csv"
]

fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    (IS_DIRECTED, "directed"),
    (HAS_VERTEX_PAIR, "has_vertex_pair"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(zip(("graph_file", "directed", "has_vertex_pair"), request.param))

    return parameters


@pytest.fixture(scope="module")
def input_expected_output(input_combo):
    """
    This fixture returns the inputs and expected results from the Sorensen algo.
    (based on cuGraph Sorensen) which can be used for validation.
    """

    input_data_path = input_combo["graph_file"]
    directed = input_combo["directed"]
    has_vertex_pair = input_combo["has_vertex_pair"]
    G = utils.generate_cugraph_graph_from_file(input_data_path, directed=directed)
    if has_vertex_pair:
        # Sample random vertices from the graph and compute the two_hop_neighbors
        # with those seeds
        k = random.randint(1, 10)
        seeds = random.sample(range(G.number_of_vertices()), k)

        vertex_pair = G.get_two_hop_neighbors(start_vertices=seeds)
    else:
        vertex_pair = None

    input_combo["vertex_pair"] = vertex_pair
    sg_cugraph_sorensen = cugraph.experimental.sorensen(G, input_combo["vertex_pair"])
    # Save the results back to the input_combo dictionary to prevent redundant
    # cuGraph runs. Other tests using the input_combo fixture will look for
    # them, and if not present they will have to re-run the same cuGraph call.

    input_combo["sg_cugraph_results"] = sg_cugraph_sorensen
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
def test_dask_sorensen(dask_client, benchmark, input_expected_output):

    dg = input_expected_output["MGGraph"]

    result_sorensen = benchmark(dcg.sorensen, dg, input_expected_output["vertex_pair"])

    result_sorensen = (
        result_sorensen.compute()
        .sort_values(["first", "second"])
        .reset_index(drop=True)
        .rename(columns={"sorensen_coeff": "mg_cugraph_sorensen_coeff"})
    )

    expected_output = (
        input_expected_output["sg_cugraph_results"]
        .sort_values(["first", "second"])
        .reset_index(drop=True)
    )

    # Update the dask cugraph sorensen results with sg cugraph results for easy
    # comparison using cuDF DataFrame methods.
    result_sorensen["sg_cugraph_sorensen_coeff"] = expected_output["sorensen_coeff"]

    sorensen_coeff_diffs1 = result_sorensen.query(
        "mg_cugraph_sorensen_coeff - sg_cugraph_sorensen_coeff > 0.00001"
    )
    sorensen_coeff_diffs2 = result_sorensen.query(
        "mg_cugraph_sorensen_coeff - sg_cugraph_sorensen_coeff < -0.00001"
    )

    assert len(sorensen_coeff_diffs1) == 0
    assert len(sorensen_coeff_diffs2) == 0


def test_dask_weighted_sorensen():
    input_data_path = datasets[0]
    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    dg = cugraph.Graph(directed=False)
    dg.from_dask_cudf_edgelist(
        ddf,
        source="src",
        destination="dst",
        edge_attr="value",
        renumber=True,
        legacy_renum_only=True,
        store_transposed=True,
    )
    with pytest.raises(ValueError):
        dcg.sorensen(dg)

    dg = cugraph.Graph(directed=False)
    dg.from_dask_cudf_edgelist(
        ddf,
        source="src",
        destination="dst",
        edge_attr="value",
        legacy_renum_only=True,
        store_transposed=True,
    )

    use_weight = True
    with pytest.raises(ValueError):
        dcg.sorensen(dg, use_weight=use_weight)
