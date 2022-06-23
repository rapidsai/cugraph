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
import cugraph.dask as dcg
import dask_cudf
from cugraph.testing import utils
import cugraph
import random


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Pytest fixtures
# =============================================================================
IS_DIRECTED = [True, False]

datasets = utils.DATASETS_UNDIRECTED + utils.DATASETS_UNRENUMBERED

fixture_params = utils.genFixtureParamsProduct(
    (datasets, "graph_file"),
    (IS_DIRECTED, "directed"),
    ([True, False], "legacy_renum_only")
    )


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(zip(("graph_file",
                           "directed",
                           "legacy_renum_only"), request.param))

    input_data_path = parameters["graph_file"]
    directed = parameters["directed"]
    legacy_renum_only = parameters["legacy_renum_only"]

    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )
    parameters["input_df"] = ddf

    dg = cugraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(
        ddf, source='src', destination='dst', edge_attr='value')

    dg.compute_renumber_edge_list(legacy_renum_only=legacy_renum_only)

    parameters["MGGraph"] = dg

    return parameters


def test_nodes_functionality(dask_client, input_combo):
    G = input_combo["MGGraph"]
    ddf = input_combo["input_df"]

    # Series has no attributed sort_values so convert the Series
    # to a DataFrame
    nodes = G.nodes().to_frame()
    col_name = nodes.columns[0]
    nodes = nodes.rename(columns={col_name: "result_nodes"})

    result_nodes = nodes.compute().sort_values(
        "result_nodes").reset_index(drop=True)

    expected_nodes = dask_cudf.concat(
        [ddf["src"], ddf["dst"]]).drop_duplicates().to_frame().sort_values(0)

    expected_nodes = expected_nodes.compute().reset_index(drop=True)

    result_nodes["expected_nodes"] = expected_nodes[0]

    compare = result_nodes.query('result_nodes != expected_nodes')

    assert len(compare) == 0


def test_has_node_functionality(dask_client, input_combo):

    G = input_combo["MGGraph"]

    valid_nodes = G.nodes().compute()

    # randomly sample k nodes from the graph
    k = random.randint(1, 20)
    n = valid_nodes.sample(k).reset_index(drop=True)
    print("nodes are \n", n)

    assert G.has_node(n)

    invalid_node = valid_nodes.max() + 1

    assert G.has_node(invalid_node) is False
