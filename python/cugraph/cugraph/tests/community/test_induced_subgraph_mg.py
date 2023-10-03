# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
from cudf.testing.testing import assert_frame_equal
import dask_cudf
import cugraph
import cugraph.dask as dcg
from cugraph.testing import utils
from cugraph.dask.common.mg_utils import is_single_gpu
from pylibcugraph.testing import gen_fixture_params_product


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


IS_DIRECTED = [True, False]
NUM_SEEDS = [2, 5, 10, 20]

# FIXME: This parameter will be tested in the next release when updating the
# SG implementation
OFFSETS = [None]


# =============================================================================
# Pytest fixtures
# =============================================================================

datasets = utils.DATASETS_UNDIRECTED + [
    utils.RAPIDS_DATASET_ROOT_DIR_PATH / "email-Eu-core.csv"
]

fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    (IS_DIRECTED, "directed"),
    (NUM_SEEDS, "num_seeds"),
    (OFFSETS, "offsets"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(
        zip(("graph_file", "directed", "seeds", "offsets"), request.param)
    )

    return parameters


@pytest.fixture(scope="module")
def input_expected_output(input_combo):
    """
    This fixture returns the inputs and expected results from the induced_subgraph algo.
    (based on cuGraph subgraph) which can be used for validation.
    """

    input_data_path = input_combo["graph_file"]
    directed = input_combo["directed"]
    num_seeds = input_combo["seeds"]

    # FIXME: This parameter is not tested
    # offsets= input_combo["offsets"]
    G = utils.generate_cugraph_graph_from_file(
        input_data_path, directed=directed, edgevals=True
    )

    # Sample k vertices from the cuGraph graph
    # FIXME: Leverage the method 'select_random_vertices' instead
    srcs = G.view_edge_list()["0"]
    dsts = G.view_edge_list()["1"]
    vertices = cudf.concat([srcs, dsts]).drop_duplicates()
    vertices = vertices.sample(num_seeds, replace=True).astype("int32")

    # print randomly sample n seeds from the graph
    print("\nvertices: \n", vertices)

    input_combo["vertices"] = vertices

    sg_induced_subgraph, _ = cugraph.induced_subgraph(G, vertices=vertices)

    # Save the results back to the input_combo dictionary to prevent redundant
    # cuGraph runs. Other tests using the input_combo fixture will look for
    # them, and if not present they will have to re-run the same cuGraph call.

    input_combo["sg_cugraph_results"] = sg_induced_subgraph
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
        store_transposed=True,
    )

    input_combo["MGGraph"] = dg

    return input_combo


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
def test_mg_induced_subgraph(dask_client, benchmark, input_expected_output):

    dg = input_expected_output["MGGraph"]
    vertices = input_expected_output["vertices"]

    result_induced_subgraph = benchmark(
        dcg.induced_subgraph,
        dg,
        vertices,
        input_expected_output["offsets"],
    )

    mg_df, mg_offsets = result_induced_subgraph

    # mg_offsets = mg_offsets.compute().reset_index(drop=True)

    sg = input_expected_output["sg_cugraph_results"]

    if mg_df is not None and sg is not None:
        # FIXME: 'edges()' or 'view_edgelist()' takes half the edges out if
        # 'directed=False'.
        sg_result = sg.input_df

        sg_df = sg_result.sort_values(["src", "dst"]).reset_index(drop=True)
        mg_df = mg_df.compute().sort_values(["src", "dst"]).reset_index(drop=True)

        assert_frame_equal(sg_df, mg_df, check_dtype=False, check_like=True)

    else:
        # There is no edges between the vertices provided
        # FIXME: Once k-hop neighbors is implemented, find one hop neighbors
        # of all the vertices and ensure that there is None
        assert sg is None
        assert mg_df is None
