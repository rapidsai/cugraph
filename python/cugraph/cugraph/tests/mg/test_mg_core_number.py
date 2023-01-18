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

import pytest
import dask_cudf
from pylibcugraph.testing.utils import gen_fixture_params_product

import cugraph
from cugraph.testing import utils
import cugraph.dask as dcg


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Pytest fixtures
# =============================================================================
datasets = utils.DATASETS_UNDIRECTED
degree_type = ["incoming", "outgoing", "bidirectional"]

fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    (degree_type, "degree_type"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(zip(("graph_file", "degree_type"), request.param))

    return parameters


@pytest.fixture(scope="module")
def input_expected_output(dask_client, input_combo):
    """
    This fixture returns the inputs and expected results from the Core number
    algo.
    """
    degree_type = input_combo["degree_type"]
    input_data_path = input_combo["graph_file"]
    G = utils.generate_cugraph_graph_from_file(
        input_data_path, directed=False, edgevals=True
    )

    input_combo["SGGraph"] = G

    sg_core_number_results = cugraph.core_number(G, degree_type)
    sg_core_number_results = sg_core_number_results.sort_values("vertex").reset_index(
        drop=True
    )

    input_combo["sg_core_number_results"] = sg_core_number_results
    input_combo["degree_type"] = degree_type

    # Creating an edgelist from a dask cudf dataframe
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
    )

    input_combo["MGGraph"] = dg

    return input_combo


# =============================================================================
# Tests
# =============================================================================
def test_sg_core_number(dask_client, benchmark, input_expected_output):
    # This test is only for benchmark purposes.
    sg_core_number_results = None
    G = input_expected_output["SGGraph"]
    degree_type = input_expected_output["degree_type"]

    sg_core_number_results = benchmark(cugraph.core_number, G, degree_type)
    assert sg_core_number_results is not None


def test_core_number(dask_client, benchmark, input_expected_output):

    dg = input_expected_output["MGGraph"]
    degree_type = input_expected_output["degree_type"]

    result_core_number = benchmark(dcg.core_number, dg, degree_type)

    result_core_number = (
        result_core_number.drop_duplicates()
        .compute()
        .sort_values("vertex")
        .reset_index(drop=True)
        .rename(columns={"core_number": "mg_core_number"})
    )

    expected_output = input_expected_output["sg_core_number_results"]

    # Update the mg core number with sg core number results
    # for easy comparison using cuDF DataFrame methods.
    result_core_number["sg_core_number"] = expected_output["core_number"]
    counts_diffs = result_core_number.query("mg_core_number != sg_core_number")

    assert len(counts_diffs) == 0


def test_core_number_invalid_input(input_expected_output):
    input_data_path = (
        utils.RAPIDS_DATASET_ROOT_DIR_PATH / "karate-asymmetric.csv"
    ).as_posix()

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
        ddf,
        source="src",
        destination="dst",
        edge_attr="value",
        renumber=True,
        legacy_renum_only=True,
    )

    invalid_degree_type = 3
    dg = input_expected_output["MGGraph"]
    with pytest.raises(ValueError):
        dcg.core_number(dg, invalid_degree_type)
