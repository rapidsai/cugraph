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

import random
import gc

import pytest
import cudf
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
fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    ([True, False], "start_list"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(zip(("graph_file", "start_list", "edgevals"), request.param))

    return parameters


@pytest.fixture(scope="module")
def input_expected_output(dask_client, input_combo):
    """
    This fixture returns the inputs and expected results from the triangle
    count algo.
    """
    start_list = input_combo["start_list"]
    input_data_path = input_combo["graph_file"]
    G = utils.generate_cugraph_graph_from_file(
        input_data_path, directed=False, edgevals=True
    )

    input_combo["SGGraph"] = G

    if start_list:
        # sample k nodes from the cuGraph graph
        k = random.randint(1, 10)
        srcs = G.view_edge_list()["src"]
        dsts = G.view_edge_list()["dst"]
        nodes = cudf.concat([srcs, dsts]).drop_duplicates()
        start_list = nodes.sample(k)
    else:
        start_list = None

    sg_triangle_results = cugraph.triangle_count(G, start_list)
    sg_triangle_results = sg_triangle_results.sort_values("vertex").reset_index(
        drop=True
    )

    input_combo["sg_triangle_results"] = sg_triangle_results
    input_combo["start_list"] = start_list

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
        ddf, source="src", destination="dst", edge_attr="value", renumber=True
    )

    input_combo["MGGraph"] = dg

    return input_combo


# =============================================================================
# Tests
# =============================================================================
def test_sg_triangles(dask_client, benchmark, input_expected_output):
    # This test is only for benchmark purposes.
    sg_triangle_results = None
    G = input_expected_output["SGGraph"]
    start_list = input_expected_output["start_list"]
    sg_triangle_results = benchmark(cugraph.triangle_count, G, start_list)
    assert sg_triangle_results is not None


def test_triangles(dask_client, benchmark, input_expected_output):

    dg = input_expected_output["MGGraph"]
    start_list = input_expected_output["start_list"]

    result_counts = benchmark(dcg.triangle_count, dg, start_list)

    result_counts = (
        result_counts.drop_duplicates()
        .compute()
        .sort_values("vertex")
        .reset_index(drop=True)
        .rename(columns={"counts": "mg_counts"})
    )

    expected_output = input_expected_output["sg_triangle_results"]

    # Update the mg triangle count with sg triangle count results
    # for easy comparison using cuDF DataFrame methods.
    result_counts["sg_counts"] = expected_output["counts"]
    counts_diffs = result_counts.query("mg_counts != sg_counts")

    assert len(counts_diffs) == 0
