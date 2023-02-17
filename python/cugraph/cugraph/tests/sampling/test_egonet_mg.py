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
from cudf.testing.testing import assert_frame_equal, assert_series_equal
from pylibcugraph.testing import gen_fixture_params_product

import cugraph
import cugraph.dask as dcg
from cugraph.testing import utils
from cugraph.dask.common.mg_utils import is_single_gpu


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


IS_DIRECTED = [True, False]
SEEDS = [0, 5, 13, [0, 2]]
RADIUS = [1, 2, 3]


# =============================================================================
# Pytest fixtures
# =============================================================================

datasets = utils.DATASETS_UNDIRECTED + [
    utils.RAPIDS_DATASET_ROOT_DIR_PATH / "email-Eu-core.csv"
]

fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    (IS_DIRECTED, "directed"),
    (SEEDS, "seeds"),
    (RADIUS, "radius"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(zip(("graph_file", "directed", "seeds", "radius"), request.param))

    return parameters


@pytest.fixture(scope="module")
def input_expected_output(input_combo):
    """
    This fixture returns the inputs and expected results from the egonet algo.
    (based on cuGraph batched_ego_graphs) which can be used for validation.
    """

    input_data_path = input_combo["graph_file"]
    directed = input_combo["directed"]
    seeds = input_combo["seeds"]
    radius = input_combo["radius"]
    G = utils.generate_cugraph_graph_from_file(input_data_path, directed=directed)

    sg_cugraph_ego_graphs = cugraph.batched_ego_graphs(G, seeds=seeds, radius=radius)

    # Save the results back to the input_combo dictionary to prevent redundant
    # cuGraph runs. Other tests using the input_combo fixture will look for
    # them, and if not present they will have to re-run the same cuGraph call.

    input_combo["sg_cugraph_results"] = sg_cugraph_ego_graphs
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


@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
def test_dask_ego_graphs(dask_client, benchmark, input_expected_output):

    dg = input_expected_output["MGGraph"]

    result_ego_graph = benchmark(
        dcg.ego_graph,
        dg,
        input_expected_output["seeds"],
        input_expected_output["radius"],
    )

    mg_df, mg_offsets = result_ego_graph

    mg_df = mg_df.compute()
    mg_offsets = mg_offsets.compute().reset_index(drop=True)

    sg_df, sg_offsets = input_expected_output["sg_cugraph_results"]

    assert_series_equal(sg_offsets, mg_offsets, check_dtype=False)
    # slice array from offsets, sort the df by src dst and compare
    for i in range(len(sg_offsets) - 1):
        start = sg_offsets[i]
        end = sg_offsets[i + 1]
        mg_df_part = mg_df[start:end].sort_values(["src", "dst"]).reset_index(drop=True)
        sg_df_part = sg_df[start:end].sort_values(["src", "dst"]).reset_index(drop=True)

        assert_frame_equal(mg_df_part, sg_df_part, check_dtype=False, check_like=True)
