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
import random
import copy
import pytest
import cupy
from dask.distributed import wait
import cudf
import dask_cudf
from pylibcugraph import bfs as pylibcugraph_bfs
from pylibcugraph import ResourceHandle
from pylibcugraph.testing.utils import gen_fixture_params_product

import cugraph
import cugraph.dask as dcg
from cugraph.testing import utils
from cugraph.dask.traversal.bfs import convert_to_cudf
import cugraph.dask.comms.comms as Comms
from cugraph.dask.common.input_utils import get_distributed_data


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

fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    (IS_DIRECTED, "directed"),
    ([True, False], "legacy_renum_only"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(
        zip(("graph_file", "directed", "legacy_renum_only"), request.param)
    )

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
        ddf,
        source="src",
        destination="dst",
        edge_attr="value",
        legacy_renum_only=legacy_renum_only,
    )

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

    result_nodes = nodes.compute().sort_values("result_nodes").reset_index(drop=True)

    expected_nodes = (
        dask_cudf.concat([ddf["src"], ddf["dst"]])
        .drop_duplicates()
        .to_frame()
        .sort_values(0)
    )

    expected_nodes = expected_nodes.compute().reset_index(drop=True)

    result_nodes["expected_nodes"] = expected_nodes[0]

    compare = result_nodes.query("result_nodes != expected_nodes")

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
    G = input_combo["MGGraph"]
    ddf = input_combo["input_df"]
    df = ddf.compute()

    # ensure graph exists
    assert G._plc_graph is not None

    # ensure graph is partitioned correctly
    assert len(G._plc_graph) == len(dask_client.has_what())

    start = dask_cudf.from_cudf(cudf.Series([1], dtype="int32"), len(G._plc_graph))

    if G.renumbered:
        start = G.lookup_internal_vertex_id(start, None)
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
                False,
            ),
            Comms.get_session_id(),
            G._plc_graph[w],
            data_start.worker_to_parts[w][0],
            workers=[w],
        )
        for w in Comms.get_workers()
    ]

    wait(res)

    cudf_result = [dask_client.submit(convert_to_cudf, cp_arrays) for cp_arrays in res]
    wait(cudf_result)

    result_dist = dask_cudf.from_delayed(cudf_result)

    if G.renumbered:
        result_dist = G.unrenumber(result_dist, "vertex")
        result_dist = G.unrenumber(result_dist, "predecessor")
        result_dist = result_dist.fillna(-1)

    result_dist = result_dist.compute()

    g = cugraph.Graph(directed=G.properties.directed)
    g.from_cudf_edgelist(df, "src", "dst")
    expected_dist = cugraph.bfs(g, cudf.Series([1], dtype="int32"))

    compare_dist = expected_dist.merge(
        result_dist, on="vertex", suffixes=["_local", "_dask"]
    )

    err = 0

    for i in range(len(compare_dist)):
        if (
            compare_dist["distance_local"].iloc[i]
            != compare_dist["distance_dask"].iloc[i]
        ):
            err = err + 1
    assert err == 0


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_create_graph_with_edge_ids(dask_client, graph_file):
    el = utils.read_csv_file(graph_file)
    el["id"] = cupy.random.permutation(len(el))
    el["id"] = el["id"].astype(el["1"].dtype)
    el["etype"] = cupy.random.random_integers(4, size=len(el))
    el["etype"] = el["etype"].astype("int32")

    num_workers = len(Comms.get_workers())
    el = dask_cudf.from_cudf(el, npartitions=num_workers)

    with pytest.raises(ValueError):
        G = cugraph.Graph()
        G.from_dask_cudf_edgelist(
            el,
            source="0",
            destination="1",
            edge_attr=["2", "id", "etype"],
        )

    G = cugraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(
        el,
        source="0",
        destination="1",
        edge_attr=["2", "id", "etype"],
        legacy_renum_only=True,
    )


def test_graph_repartition(dask_client):
    input_data_path = (utils.RAPIDS_DATASET_ROOT_DIR_PATH / "karate.csv").as_posix()
    print(f"dataset={input_data_path}")
    chunksize = dcg.get_chunksize(input_data_path)

    num_workers = len(Comms.get_workers())

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )
    more_partitions = num_workers * 100
    ddf = ddf.repartition(npartitions=more_partitions)
    ddf = get_distributed_data(ddf)

    num_futures = len(ddf.worker_to_parts.values())
    assert num_futures == num_workers


def test_mg_graph_serializable(dask_client, input_combo):
    G = input_combo["MGGraph"]
    dask_client.publish_dataset(shared_g=G)
    shared_g = dask_client.get_dataset("shared_g")
    assert type(shared_g) == type(G)
    assert G.number_of_vertices() == shared_g.number_of_vertices()
    assert G.number_of_edges() == shared_g.number_of_edges()
    # cleanup
    dask_client.unpublish_dataset("shared_g")


def test_mg_graph_copy():
    G = cugraph.MultiGraph(directed=True)
    G_c = copy.deepcopy(G)
    assert type(G) == type(G_c)
