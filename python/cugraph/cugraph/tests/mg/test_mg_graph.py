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

from pylibcugraph import bfs as pylibcugraph_bfs
from pylibcugraph import ResourceHandle

from cugraph.dask.traversal.bfs import convert_to_cudf

import cugraph.dask.comms.comms as Comms
from cugraph.dask.common.input_utils import get_distributed_data
from dask.distributed import wait
import cudf


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
    ([True], "legacy_renum_only")
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
        ddf, source='src', destination='dst', edge_attr='value',
        legacy_renum_only=legacy_renum_only)

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


def test_create_mg_graph(dask_client, input_combo):
    G = input_combo['MGGraph']

    # ensure graph exists
    assert G._plc_graph is not None

    # ensure graph is partitioned correctly
    assert len(G._plc_graph) == len(dask_client.has_what())

    start = dask_cudf.from_cudf(
        cudf.Series([1], dtype='int32'),
        len(G._plc_graph)
    )
    data_start = get_distributed_data(start)

    res = [
        dask_client.submit(
            lambda sID, mg_graph_x, st_x: pylibcugraph_bfs(
                ResourceHandle(Comms.get_handle(sID).getHandle()),
                mg_graph_x,
                st_x,
                False,
                0,
                True,
                False
            ),
            Comms.get_session_id(),
            G._plc_graph[w],
            data_start.worker_to_parts[w][0],
            workers=[w]
        )
        for w in Comms.get_workers()
    ]

    wait(res)

    cudf_result = [
        dask_client.submit(convert_to_cudf, cp_arrays)
        for cp_arrays in res
    ]
    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result)

    ddf = ddf.compute()

    if 'dolphins.csv' == input_combo['graph_file'].name:
        assert ddf[ddf.vertex == 33].distance.iloc[0] == 3
        assert ddf[ddf.vertex == 33].predecessor.iloc[0] == 37
        assert ddf[ddf.vertex == 11].distance.iloc[0] == 4
        assert ddf[ddf.vertex == 11].predecessor.iloc[0] == 51
    else:
        assert ddf[ddf.vertex == 33].distance.iloc[0] == 2
        assert ddf[ddf.vertex == 33].predecessor.iloc[0] == 30
        assert ddf[ddf.vertex == 11].distance.iloc[0] == 2
        assert ddf[ddf.vertex == 11].predecessor.iloc[0] == 0
