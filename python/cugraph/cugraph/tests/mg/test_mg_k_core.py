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
from cudf.testing.testing import assert_frame_equal
from pylibcugraph.testing import gen_fixture_params_product

import cugraph
from cugraph.testing import utils
import cugraph.dask as dcg
from cugraph.structure.symmetrize import symmetrize_df


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Pytest fixtures
# =============================================================================
datasets = utils.DATASETS_UNDIRECTED

core_number = [True, False]
degree_type = ["bidirectional", "outgoing", "incoming"]

fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"), (core_number, "core_number"), (degree_type, "degree_type")
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(zip(("graph_file", "core_number", "degree_type"), request.param))

    return parameters


@pytest.fixture(scope="module")
def input_expected_output(dask_client, input_combo):
    """
    This fixture returns the inputs and expected results from the Core number
    algo.
    """
    core_number = input_combo["core_number"]
    degree_type = input_combo["degree_type"]
    input_data_path = input_combo["graph_file"]
    G = utils.generate_cugraph_graph_from_file(
        input_data_path, directed=False, edgevals=True
    )

    if core_number:
        # compute the core_number
        core_number = cugraph.core_number(G, degree_type=degree_type)
    else:
        core_number = None

    input_combo["core_number"] = core_number

    input_combo["SGGraph"] = G

    sg_k_core_graph = cugraph.k_core(
        G, core_number=core_number, degree_type=degree_type
    )
    sg_k_core_results = sg_k_core_graph.view_edge_list()
    # FIXME: The result will come asymetric. Symmetrize the results
    sg_k_core_results = (
        symmetrize_df(sg_k_core_results, "src", "dst", "weights")
        .sort_values(["src", "dst"])
        .reset_index(drop=True)
    )

    input_combo["sg_k_core_results"] = sg_k_core_results

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
    # FIXME: False when renumbering (C++ and python renumbering)
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
def test_sg_k_core(dask_client, benchmark, input_expected_output):
    # This test is only for benchmark purposes.
    sg_k_core = None
    G = input_expected_output["SGGraph"]
    core_number = input_expected_output["core_number"]
    degree_type = input_expected_output["degree_type"]

    sg_k_core = benchmark(
        cugraph.k_core, G, core_number=core_number, degree_type=degree_type
    )
    assert sg_k_core is not None


def test_dask_k_core(dask_client, benchmark, input_expected_output):

    dg = input_expected_output["MGGraph"]
    core_number = input_expected_output["core_number"]

    k_core_results = benchmark(dcg.k_core, dg, core_number=core_number)

    expected_k_core_results = input_expected_output["sg_k_core_results"]

    k_core_results = (
        k_core_results.compute().sort_values(["src", "dst"]).reset_index(drop=True)
    )

    assert_frame_equal(
        expected_k_core_results, k_core_results, check_dtype=False, check_like=True
    )


def test_dask_k_core_invalid_input(dask_client):
    input_data_path = datasets[0]
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
        store_transposed=True,
    )
    with pytest.raises(ValueError):
        dcg.k_core(dg)

    dg = cugraph.Graph(directed=False)
    dg.from_dask_cudf_edgelist(
        ddf,
        source="src",
        destination="dst",
        edge_attr="value",
        legacy_renum_only=True,
        store_transposed=True,
    )

    degree_type = "invalid"
    with pytest.raises(ValueError):
        dcg.k_core(dg, degree_type=degree_type)
