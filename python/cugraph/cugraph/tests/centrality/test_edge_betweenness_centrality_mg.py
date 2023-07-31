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

import dask_cudf
from pylibcugraph.testing.utils import gen_fixture_params_product
from cugraph.experimental.datasets import DATASETS_UNDIRECTED, email_Eu_core

import cugraph
import cugraph.dask as dcg

# from cugraph.dask.common.mg_utils import is_single_gpu


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


IS_DIRECTED = [True, False]
INCLUDE_WEIGHTS = [False, True]
INCLUDE_EDGE_IDS = [False, True]
NORMALIZED_OPTIONS = [False, True]
SUBSET_SIZE_OPTIONS = [4, None]


# email_Eu_core is too expensive to test
datasets = DATASETS_UNDIRECTED + [email_Eu_core]


# =============================================================================
# Pytest fixtures
# =============================================================================


fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    (IS_DIRECTED, "directed"),
    (INCLUDE_WEIGHTS, "include_weights"),
    (INCLUDE_EDGE_IDS, "include_edgeids"),
    (NORMALIZED_OPTIONS, "normalized"),
    (SUBSET_SIZE_OPTIONS, "subset_size"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(
        zip(
            (
                "graph_file",
                "directed",
                "include_weights",
                "include_edge_ids",
                "normalized",
                "subset_size",
                "subset_seed",
            ),
            request.param,
        )
    )

    return parameters


@pytest.fixture(scope="module")
def input_expected_output(input_combo):
    """
    This fixture returns the inputs and expected results from the edge
    betweenness centrality algo.
    (based on cuGraph edge betweenness centrality) which can be used
    for validation.
    """
    directed = input_combo["directed"]
    normalized = input_combo["normalized"]
    k = input_combo["subset_size"]
    subset_seed = 42
    edge_ids = input_combo["include_edge_ids"]
    weight = input_combo["include_weights"]

    df = input_combo["graph_file"].get_edgelist()
    if edge_ids:
        if not directed:
            # Edge ids not supported for undirected graph
            return
        dtype = df.dtypes[0]
        edge_id = "edge_id"
        df["edge_id"] = df.index
        df = df.astype(dtype)

    else:
        edge_id = None

    G = cugraph.Graph(directed=directed)
    G.from_cudf_edgelist(
        df, source="src", destination="dst", weight="wgt", edge_id=edge_id
    )
    if isinstance(k, int):
        k = G.select_random_vertices(subset_seed, k)

    input_combo["k"] = k
    # Save the results back to the input_combo dictionary to prevent redundant
    # cuGraph runs. Other tests using the input_combo fixture will look for
    # them, and if not present they will have to re-run the same cuGraph call.
    sg_cugraph_edge_bc = (
        cugraph.edge_betweenness_centrality(G, k, normalized)
        .sort_values(["src", "dst"])
        .reset_index(drop=True)
    )

    input_data_path = input_combo["graph_file"].get_path()

    input_combo["sg_cugraph_results"] = sg_cugraph_edge_bc
    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    if weight:
        weight = ddf
    else:
        weight = None

    if edge_ids:
        dtype = ddf.dtypes[0]
        edge_id = "edge_id"
        ddf = ddf.assign(idx=1)
        ddf["edge_id"] = ddf.idx.cumsum().astype(dtype) - 1
    else:
        edge_id = None

    dg = cugraph.Graph(directed=directed)

    dg.from_dask_cudf_edgelist(
        ddf,
        source="src",
        destination="dst",
        weight="value",
        edge_id=edge_id,
        renumber=True,
    )

    input_combo["MGGraph"] = dg
    input_combo["include_weights"] = weight

    return input_combo


# =============================================================================
# Tests
# =============================================================================


# @pytest.mark.skipif(
#    is_single_gpu(), reason="skipping MG testing on Single GPU system"
# )
@pytest.mark.mg
def test_dask_edge_betweenness_centrality(
    dask_client, benchmark, input_expected_output
):
    if input_expected_output is not None:
        dg = input_expected_output["MGGraph"]
        k = input_expected_output["k"]
        normalized = input_expected_output["normalized"]
        weight = input_expected_output["include_weights"]
        if weight is not None:
            with pytest.raises(NotImplementedError):
                result_edge_bc = benchmark(
                    dcg.edge_betweenness_centrality, dg, k, normalized, weight=weight
                )

        else:
            result_edge_bc = benchmark(
                dcg.edge_betweenness_centrality, dg, k, normalized, weight=weight
            )
            result_edge_bc = (
                result_edge_bc.compute()
                .sort_values(["src", "dst"])
                .reset_index(drop=True)
                .rename(columns={"betweenness_centrality": "mg_betweenness_centrality"})
            )

            if len(result_edge_bc.columns) > 3:
                result_edge_bc = result_edge_bc.rename(
                    columns={"edge_id": "mg_edge_id"}
                )

            expected_output = input_expected_output["sg_cugraph_results"].reset_index(
                drop=True
            )
            result_edge_bc["betweenness_centrality"] = expected_output[
                "betweenness_centrality"
            ]
            if len(expected_output.columns) > 3:
                result_edge_bc["edge_id"] = expected_output["edge_id"]
                edge_id_diff = result_edge_bc.query("mg_edge_id != edge_id")
                assert len(edge_id_diff) == 0

            edge_bc_diffs1 = result_edge_bc.query(
                "mg_betweenness_centrality - betweenness_centrality > 0.01"
            )
            edge_bc_diffs2 = result_edge_bc.query(
                "betweenness_centrality - mg_betweenness_centrality < -0.01"
            )

            assert len(edge_bc_diffs1) == 0
            assert len(edge_bc_diffs2) == 0
