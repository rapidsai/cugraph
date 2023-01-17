# Copyright (c) 2022-2023, NVIDIA CORPORATION.
#
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

import importlib
import random

import pytest
import pandas as pd
import numpy as np
import cupy
import cudf
import cugraph
from cugraph.experimental import PropertyGraph

from cugraph_service_client import RemoteGraph
from . import data


# FIXME: Remove this once these pass in the CI environment.
pytest.skip(
    reason="FIXME: many of these tests fail in CI and are currently run "
    "manually only in dev environments.",
    allow_module_level=True,
)

###############################################################################
# fixtures
# The fixtures used in these tests are defined here and in conftest.py


@pytest.fixture(scope="function")
def client_with_property_csvs_loaded(client):
    """
    Loads each of the vertex and edge property CSVs into the default graph on
    the server.
    """

    merchants = data.property_csv_data["merchants"]
    users = data.property_csv_data["users"]
    transactions = data.property_csv_data["transactions"]
    relationships = data.property_csv_data["relationships"]
    referrals = data.property_csv_data["referrals"]

    client.load_csv_as_vertex_data(
        merchants["csv_file_name"],
        dtypes=merchants["dtypes"],
        vertex_col_name=merchants["vert_col_name"],
        header=0,
        type_name="merchants",
    )
    client.load_csv_as_vertex_data(
        users["csv_file_name"],
        dtypes=users["dtypes"],
        vertex_col_name=users["vert_col_name"],
        header=0,
        type_name="users",
    )

    client.load_csv_as_edge_data(
        transactions["csv_file_name"],
        dtypes=transactions["dtypes"],
        vertex_col_names=transactions["vert_col_names"],
        header=0,
        type_name="transactions",
    )
    client.load_csv_as_edge_data(
        relationships["csv_file_name"],
        dtypes=relationships["dtypes"],
        vertex_col_names=relationships["vert_col_names"],
        header=0,
        type_name="relationships",
    )
    client.load_csv_as_edge_data(
        referrals["csv_file_name"],
        dtypes=referrals["dtypes"],
        vertex_col_names=referrals["vert_col_names"],
        header=0,
        type_name="referrals",
    )

    assert client.get_graph_ids() == [0]
    return client


@pytest.fixture(scope="function")
def pG_with_property_csvs_loaded():
    """
    Loads each of the vertex and edge property CSVs into a
    property graph.
    """
    pG = PropertyGraph()
    merchants = data.property_csv_data["merchants"]
    users = data.property_csv_data["users"]
    transactions = data.property_csv_data["transactions"]
    relationships = data.property_csv_data["relationships"]
    referrals = data.property_csv_data["referrals"]

    merchants_df = cudf.read_csv(
        merchants["csv_file_name"], dtype=merchants["dtypes"], header=0, delimiter=" "
    )
    pG.add_vertex_data(
        merchants_df,
        vertex_col_name=merchants["vert_col_name"],
        type_name="merchants",
    )

    users_df = cudf.read_csv(
        users["csv_file_name"], dtype=users["dtypes"], header=0, delimiter=" "
    )
    pG.add_vertex_data(
        users_df,
        vertex_col_name=users["vert_col_name"],
        type_name="users",
    )

    transactions_df = cudf.read_csv(
        transactions["csv_file_name"],
        dtype=transactions["dtypes"],
        header=0,
        delimiter=" ",
    )
    pG.add_edge_data(
        transactions_df,
        vertex_col_names=transactions["vert_col_names"],
        type_name="transactions",
    )

    relationships_df = cudf.read_csv(
        relationships["csv_file_name"],
        dtype=relationships["dtypes"],
        header=0,
        delimiter=" ",
    )
    pG.add_edge_data(
        relationships_df,
        vertex_col_names=relationships["vert_col_names"],
        type_name="relationships",
    )

    referrals_df = cudf.read_csv(
        referrals["csv_file_name"], dtype=referrals["dtypes"], header=0, delimiter=" "
    )
    pG.add_edge_data(
        referrals_df,
        vertex_col_names=referrals["vert_col_names"],
        type_name="referrals",
    )
    return pG


def test_graph_info(client_with_property_csvs_loaded, pG_with_property_csvs_loaded):
    rpG = RemoteGraph(client_with_property_csvs_loaded, 0)
    pG = pG_with_property_csvs_loaded
    graph_info = rpG.graph_info

    expected_results = {
        "num_edges": pG.get_num_edges(),
        "num_edge_properties": len(pG.edge_property_names),
        "num_vertices": pG.get_num_vertices(),
        "num_vertex_properties": len(pG.vertex_property_names),
        "num_vertices_from_vertex_data": pG.get_num_vertices(include_edge_data=False),
        "is_multi_gpu": False,
    }

    assert set(graph_info.keys()) == set(expected_results.keys())
    for k in expected_results:
        assert graph_info[k] == expected_results[k]


def test_edges(client_with_property_csvs_loaded, pG_with_property_csvs_loaded):
    # FIXME update this when edges() method issue is resolved.
    rpG = RemoteGraph(client_with_property_csvs_loaded, 0)
    pG = pG_with_property_csvs_loaded

    edges = pG.get_edge_data(
        columns=[pG.src_col_name, pG.dst_col_name, pG.type_col_name]
    )
    rpG_edges = rpG.edges()

    assert (edges[pG.edge_id_col_name] == rpG_edges[rpG.edge_id_col_name]).all()
    assert (edges[pG.src_col_name] == rpG_edges[rpG.src_col_name]).all()
    assert (edges[pG.dst_col_name] == rpG_edges[rpG.dst_col_name]).all()
    assert (
        edges[pG.type_col_name].astype("string")
        == rpG_edges[rpG.type_col_name].astype("string")
    ).all()


def test_property_type_names(
    client_with_property_csvs_loaded, pG_with_property_csvs_loaded
):
    rpG = RemoteGraph(client_with_property_csvs_loaded, 0)
    pG = pG_with_property_csvs_loaded

    assert rpG.vertex_property_names == pG.vertex_property_names
    assert rpG.edge_property_names == pG.edge_property_names
    assert rpG.vertex_types == pG.vertex_types
    assert rpG.edge_types == pG.edge_types


def test_num_elements(client_with_property_csvs_loaded, pG_with_property_csvs_loaded):
    rpG = RemoteGraph(client_with_property_csvs_loaded, 0)
    pG = pG_with_property_csvs_loaded

    assert rpG.get_num_vertices() == pG.get_num_vertices()
    assert rpG.get_num_vertices(include_edge_data=False) == pG.get_num_vertices(
        include_edge_data=False
    )
    for type in pG.vertex_types:
        assert rpG.get_num_vertices(type=type) == pG.get_num_vertices(type=type)
        assert rpG.get_num_vertices(
            type=type, include_edge_data=False
        ) == pG.get_num_vertices(type=type, include_edge_data=False)

    assert rpG.get_num_edges() == pG.get_num_edges()
    for type in pG.edge_types:
        assert rpG.get_num_edges(type=type) == pG.get_num_edges(type=type)


def test_get_vertex_data(
    client_with_property_csvs_loaded, pG_with_property_csvs_loaded
):
    rpG = RemoteGraph(client_with_property_csvs_loaded, 0)
    pG = pG_with_property_csvs_loaded

    vd = rpG.get_vertex_data()
    vd[rpG.type_col_name] = vd[rpG.type_col_name].astype("string")
    expected_vd = pG.get_vertex_data().fillna(0)  # FIXME expose na handling
    expected_vd[pG.type_col_name] = expected_vd[pG.type_col_name].astype("string")
    for col in expected_vd.columns:
        assert (expected_vd[col] == vd[col]).all()

    for _ in range(3):
        vertex_ids = random.sample(pG.vertices_ids().values_host.tolist(), 3)
        vd = rpG.get_vertex_data(vertex_ids=vertex_ids)
        vd[rpG.type_col_name] = vd[rpG.type_col_name].astype("string")
        expected_vd = pG.get_vertex_data(vertex_ids=vertex_ids).fillna(
            0
        )  # FIXME expose na handling
        expected_vd[pG.type_col_name] = expected_vd[pG.type_col_name].astype("string")
        for col in expected_vd.columns:
            assert (expected_vd[col] == vd[col]).all()

    vertex_type_list = [["merchants", "users"], ["merchants"]]
    for vertex_types in vertex_type_list:
        vd = rpG.get_vertex_data(types=vertex_types)
        vd[rpG.type_col_name] = vd[rpG.type_col_name].astype("string")
        expected_vd = pG.get_vertex_data(types=vertex_types).fillna(
            0
        )  # FIXME expose na handling
        expected_vd[pG.type_col_name] = expected_vd[pG.type_col_name].astype("string")
        for col in expected_vd.columns:
            assert (expected_vd[col] == vd[col]).all()

    vd = rpG.get_vertex_data(types=["users"], columns=["vertical"])
    vd[rpG.type_col_name] = vd[rpG.type_col_name].astype("string")
    expected_vd = pG.get_vertex_data(types=["users"], columns=["vertical"]).fillna(
        0
    )  # FIXME expose na handling
    expected_vd[pG.type_col_name] = expected_vd[pG.type_col_name].astype("string")
    for col in expected_vd.columns:
        assert (expected_vd[col] == vd[col]).all()


def test_get_edge_data(client_with_property_csvs_loaded, pG_with_property_csvs_loaded):
    rpG = RemoteGraph(client_with_property_csvs_loaded, 0)
    pG = pG_with_property_csvs_loaded

    ed = rpG.get_edge_data()
    ed[rpG.type_col_name] = ed[rpG.type_col_name].astype("string")
    expected_ed = pG.get_edge_data().fillna(0)  # FIXME expose na handling
    expected_ed[pG.type_col_name] = expected_ed[pG.type_col_name].astype("string")
    for col in expected_ed.columns:
        assert (expected_ed[col] == ed[col]).all()

    for _ in range(3):
        edge_ids = random.sample(
            pG.get_edge_data()[pG.edge_id_col_name].values_host.tolist(), 3
        )
        ed = rpG.get_edge_data(edge_ids=edge_ids)
        ed[rpG.type_col_name] = ed[rpG.type_col_name].astype("string")
        expected_ed = pG.get_edge_data(edge_ids=edge_ids).fillna(
            0
        )  # FIXME expose na handling
        expected_ed[pG.type_col_name] = expected_ed[pG.type_col_name].astype("string")
        for col in expected_ed.columns:
            assert (expected_ed[col] == ed[col]).all()

    for edge_types in [["transactions", "relationships"], ["referrals"]]:
        ed = rpG.get_edge_data(types=edge_types)
        ed[rpG.type_col_name] = ed[rpG.type_col_name].astype("string")
        expected_ed = pG.get_edge_data(types=edge_types).fillna(
            0
        )  # FIXME expose na handling
        expected_ed[pG.type_col_name] = expected_ed[pG.type_col_name].astype("string")
        for col in expected_ed.columns:
            assert (expected_ed[col] == ed[col]).all()

    ed = rpG.get_edge_data(types=["referrals"], columns=["stars", "merchant_id"])
    ed[rpG.type_col_name] = ed[rpG.type_col_name].astype("string")
    expected_ed = pG.get_edge_data(
        types=["referrals"], columns=["stars", "merchant_id"]
    ).fillna(
        0
    )  # FIXME expose na handling
    expected_ed[pG.type_col_name] = expected_ed[pG.type_col_name].astype("string")
    for col in expected_ed.columns:
        assert (expected_ed[col] == ed[col]).all()


@pytest.mark.skip(reason="not yet implemented")
def test_add_vertex_data(
    client_with_property_csvs_loaded, pG_with_property_csvs_loaded
):
    raise NotImplementedError()


@pytest.mark.skip(reason="not yet implemented")
def test_add_edge_data(client_with_property_csvs_loaded, pG_with_property_csvs_loaded):

    raise NotImplementedError()


def test_get_vertices(client_with_property_csvs_loaded, pG_with_property_csvs_loaded):
    rpG = RemoteGraph(client_with_property_csvs_loaded, 0)
    pG = pG_with_property_csvs_loaded

    assert set(rpG.get_vertices().to_cupy().tolist()) == set(
        pG.get_vertices().to_cupy().tolist()
    )


@pytest.mark.skip(reason="not yet implemented")
def test_get_vertices_with_selection(
    client_with_property_csvs_loaded, pG_with_property_csvs_loaded
):
    raise NotImplementedError()


@pytest.mark.parametrize(
    "create_using",
    [
        (None, None),
        (cugraph.Graph(), "Graph"),
        (cugraph.MultiGraph(), "MultiGraph"),
        (cugraph.Graph(directed=True), "Graph(directed=True)"),
        (cugraph.MultiGraph(directed=True), "MultiGraph(directed=True)"),
    ],
)
@pytest.mark.parametrize(
    "selection",
    [
        (True, None),
        (False, '_TYPE_=="transactions"'),
        (True, '(_TYPE_=="transactions") | (_TYPE_=="relationships")'),
    ],
)
@pytest.mark.parametrize("renumber", [False, True])
def test_extract_subgraph(
    client_with_property_csvs_loaded,
    pG_with_property_csvs_loaded,
    create_using,
    selection,
    renumber,
):
    mg_only, selection = selection
    if mg_only and create_using[0] is not None and not create_using[0].is_multigraph():
        pytest.skip()

    rpG = RemoteGraph(client_with_property_csvs_loaded, 0)
    pG = pG_with_property_csvs_loaded

    sg = pG.extract_subgraph(
        create_using=create_using[0],
        selection=None if selection is None else pG.select_edges(selection),
        renumber_graph=renumber,
    )
    remote_sg = rpG.extract_subgraph(
        create_using=create_using[1], selection=selection, renumber_graph=renumber
    )

    assert remote_sg.get_num_vertices() == sg.number_of_vertices()

    expected_vertex_ids = cudf.concat(
        [sg.edgelist.edgelist_df["src"], sg.edgelist.edgelist_df["dst"]]
    ).unique()
    if renumber:
        expected_vertex_ids = sg.unrenumber(
            cudf.DataFrame({"v": expected_vertex_ids}), "v"
        )["v"]
    assert set(remote_sg.vertices_ids().to_cupy().tolist()) == set(
        expected_vertex_ids.to_cupy().tolist()
    )

    expected_edgelist = sg.edgelist.edgelist_df
    if renumber:
        expected_edgelist = sg.unrenumber(expected_edgelist, "src")
        expected_edgelist = sg.unrenumber(expected_edgelist, "dst")

    expected_edgelist = expected_edgelist.sort_values(["src", "dst"])

    edge_data = remote_sg.get_edge_data().sort_values(
        [remote_sg.src_col_name, remote_sg.dst_col_name]
    )
    assert (
        expected_edgelist["src"].to_cupy().tolist()
        == edge_data[remote_sg.src_col_name].to_cupy().tolist()
    )
    assert (
        expected_edgelist["dst"].to_cupy().tolist()
        == edge_data[remote_sg.dst_col_name].to_cupy().tolist()
    )


def test_backend_pandas(client_with_property_csvs_loaded, pG_with_property_csvs_loaded):
    rpG = RemoteGraph(client_with_property_csvs_loaded, 0)
    pG = pG_with_property_csvs_loaded

    # edges()
    rpg_edges = rpG.edges(backend="pandas")
    pg_edges = pG.get_edge_data(
        columns=[pG.src_col_name, pG.dst_col_name, pG.type_col_name]
    )
    assert isinstance(rpg_edges, pd.DataFrame)
    assert (
        rpg_edges[rpG.src_col_name].tolist()
        == pg_edges[pG.src_col_name].values_host.tolist()
    )
    assert (
        rpg_edges[rpG.dst_col_name].tolist()
        == pg_edges[pG.dst_col_name].values_host.tolist()
    )
    assert (
        rpg_edges[rpG.type_col_name].tolist()
        == pg_edges[pG.type_col_name].values_host.tolist()
    )
    assert (
        rpg_edges[rpG.edge_id_col_name].tolist()
        == pg_edges[pG.edge_id_col_name].values_host.tolist()
    )

    # get_vertex_data()
    rpg_vertex_data = rpG.get_vertex_data(backend="pandas")
    pg_vertex_data = pG.get_vertex_data().fillna(0)
    assert isinstance(rpg_vertex_data, pd.DataFrame)
    assert sorted(list(rpg_vertex_data.columns)) == sorted(list(pg_vertex_data.columns))
    for col in rpg_vertex_data.columns:
        assert rpg_vertex_data[col].tolist() == pg_vertex_data[col].values_host.tolist()

    # get_edge_data()
    rpg_edge_data = rpG.get_edge_data(backend="pandas")
    pg_edge_data = pG.get_edge_data().fillna(0)
    assert isinstance(rpg_edge_data, pd.DataFrame)
    assert sorted(list(rpg_edge_data.columns)) == sorted(list(pg_edge_data.columns))
    for col in rpg_edge_data.columns:
        assert rpg_edge_data[col].tolist() == pg_edge_data[col].values_host.tolist()


def test_backend_cupy(client_with_property_csvs_loaded, pG_with_property_csvs_loaded):
    rpG = RemoteGraph(client_with_property_csvs_loaded, 0)
    pG = pG_with_property_csvs_loaded

    # edges()
    rpg_edges = rpG.edges(backend="cupy")
    pg_edges = pG.get_edge_data(
        columns=[pG.src_col_name, pG.dst_col_name, pG.type_col_name]
    )
    for out_tensor in rpg_edges:
        assert isinstance(out_tensor, cupy.ndarray)
    assert rpg_edges[1].get().tolist() == pg_edges[pG.src_col_name].values_host.tolist()
    assert rpg_edges[2].get().tolist() == pg_edges[pG.dst_col_name].values_host.tolist()
    assert (
        rpg_edges[0].get().tolist()
        == pg_edges[pG.edge_id_col_name].values_host.tolist()
    )

    rpg_types = rpg_edges[3].get().tolist()
    pg_types = [
        rpG._edge_categorical_dtype[t] for t in pg_edges[pG.type_col_name].values_host
    ]
    assert rpg_types == pg_types

    # get_vertex_data()
    cols_of_interest = [
        "merchant_location",
        "merchant_size",
        "merchant_sales",
        "merchant_num_employees",
    ]
    rpg_vertex_data = rpG.get_vertex_data(
        types=["merchants"], columns=cols_of_interest, backend="cupy"
    )
    pg_vertex_data = pG.get_vertex_data(
        types=["merchants"], columns=cols_of_interest
    ).fillna(0)
    for out_tensor in rpg_vertex_data:
        assert isinstance(out_tensor, cupy.ndarray)
    assert len(rpg_vertex_data) == len(pg_vertex_data.columns)
    for i, col in enumerate(cols_of_interest):
        assert (
            rpg_vertex_data[i + 2].tolist() == pg_vertex_data[col].values_host.tolist()
        )

    # get_edge_data()
    cols_of_interest = ["time", "volume", "card_num"]
    rpg_edge_data = rpG.get_edge_data(
        types=["transactions"], columns=cols_of_interest, backend="cupy"
    )
    pg_edge_data = pG.get_edge_data(
        types=["transactions"], columns=cols_of_interest
    ).fillna(0)
    for out_tensor in rpg_edge_data:
        assert isinstance(out_tensor, cupy.ndarray)
    assert len(rpg_edge_data) == len(pg_edge_data.columns)
    for i, col in enumerate(cols_of_interest):
        assert rpg_edge_data[i + 4].tolist() == pg_edge_data[col].values_host.tolist()


def test_backend_numpy(client_with_property_csvs_loaded, pG_with_property_csvs_loaded):
    rpG = RemoteGraph(client_with_property_csvs_loaded, 0)
    pG = pG_with_property_csvs_loaded

    # edges()
    rpg_edges = rpG.edges(backend="numpy")
    pg_edges = pG.get_edge_data(
        columns=[pG.src_col_name, pG.dst_col_name, pG.type_col_name]
    )
    for out_tensor in rpg_edges:
        assert isinstance(out_tensor, np.ndarray)
    assert rpg_edges[1].tolist() == pg_edges[pG.src_col_name].values_host.tolist()
    assert rpg_edges[2].tolist() == pg_edges[pG.dst_col_name].values_host.tolist()
    assert rpg_edges[0].tolist() == pg_edges[pG.edge_id_col_name].values_host.tolist()

    rpg_types = rpg_edges[3].tolist()
    pg_types = [
        rpG._edge_categorical_dtype[t] for t in pg_edges[pG.type_col_name].values_host
    ]
    assert rpg_types == pg_types

    # get_vertex_data()
    cols_of_interest = [
        "merchant_location",
        "merchant_size",
        "merchant_sales",
        "merchant_num_employees",
    ]
    rpg_vertex_data = rpG.get_vertex_data(
        types=["merchants"], columns=cols_of_interest, backend="numpy"
    )
    pg_vertex_data = pG.get_vertex_data(
        types=["merchants"], columns=cols_of_interest
    ).fillna(0)
    for out_tensor in rpg_vertex_data:
        assert isinstance(out_tensor, np.ndarray)
    assert len(rpg_vertex_data) == len(pg_vertex_data.columns)
    for i, col in enumerate(cols_of_interest):
        assert (
            rpg_vertex_data[i + 2].tolist() == pg_vertex_data[col].values_host.tolist()
        )

    # get_edge_data()
    cols_of_interest = ["time", "volume", "card_num"]
    rpg_edge_data = rpG.get_edge_data(
        types=["transactions"], columns=cols_of_interest, backend="numpy"
    )
    pg_edge_data = pG.get_edge_data(
        types=["transactions"], columns=cols_of_interest
    ).fillna(0)
    for out_tensor in rpg_edge_data:
        assert isinstance(out_tensor, np.ndarray)
    assert len(rpg_edge_data) == len(pg_edge_data.columns)
    for i, col in enumerate(cols_of_interest):
        assert rpg_edge_data[i + 4].tolist() == pg_edge_data[col].values_host.tolist()


try:
    torch = importlib.import_module("torch")
except ModuleNotFoundError:
    torch = None


@pytest.mark.skipif(torch is None, reason="torch not available")
@pytest.mark.parametrize("torch_backend", ["torch", "torch:0", "torch:cuda"])
def test_backend_torch(
    client_with_property_csvs_loaded, pG_with_property_csvs_loaded, torch_backend
):
    rpG = RemoteGraph(client_with_property_csvs_loaded, 0)
    pG = pG_with_property_csvs_loaded

    # edges()
    rpg_edges = rpG.edges(backend=torch_backend)
    pg_edges = pG.get_edge_data(
        columns=[pG.src_col_name, pG.dst_col_name, pG.type_col_name]
    )
    for out_tensor in rpg_edges:
        assert isinstance(out_tensor, torch.Tensor)
    assert rpg_edges[1].tolist() == pg_edges[pG.src_col_name].values_host.tolist()
    assert rpg_edges[2].tolist() == pg_edges[pG.dst_col_name].values_host.tolist()
    assert rpg_edges[0].tolist() == pg_edges[pG.edge_id_col_name].values_host.tolist()

    rpg_types = rpg_edges[3].tolist()
    pg_types = [
        rpG._edge_categorical_dtype[t] for t in pg_edges[pG.type_col_name].values_host
    ]
    assert rpg_types == pg_types

    # get_vertex_data()
    cols_of_interest = [
        "merchant_location",
        "merchant_size",
        "merchant_sales",
        "merchant_num_employees",
    ]
    rpg_vertex_data = rpG.get_vertex_data(
        types=["merchants"], columns=cols_of_interest, backend=torch_backend
    )
    pg_vertex_data = pG.get_vertex_data(
        types=["merchants"], columns=cols_of_interest
    ).fillna(0)
    for out_tensor in rpg_vertex_data:
        assert isinstance(out_tensor, torch.Tensor)
    assert len(rpg_vertex_data) == len(pg_vertex_data.columns)
    for i, col in enumerate(cols_of_interest):
        assert (
            rpg_vertex_data[i + 2].tolist() == pg_vertex_data[col].values_host.tolist()
        )

    # get_edge_data()
    cols_of_interest = ["time", "volume", "card_num"]
    rpg_edge_data = rpG.get_edge_data(
        types=["transactions"], columns=cols_of_interest, backend=torch_backend
    )
    pg_edge_data = pG.get_edge_data(
        types=["transactions"], columns=cols_of_interest
    ).fillna(0)
    for out_tensor in rpg_edge_data:
        assert isinstance(out_tensor, torch.Tensor)
    assert len(rpg_edge_data) == len(pg_edge_data.columns)
    for i, col in enumerate(cols_of_interest):
        assert rpg_edge_data[i + 4].tolist() == pg_edge_data[col].values_host.tolist()


def test_remote_graph_neighbor_sample(
    client_with_property_csvs_loaded, pG_with_property_csvs_loaded
):
    # FIXME: consider moving the call dispatcher into cugraph-service-client or
    # cugraph proper. Import it here for now to allow tests to run in an
    # environment without cugraph-pyg.
    from cugraph_pyg.loader.dispatch import call_cugraph_algorithm

    rpG = RemoteGraph(client_with_property_csvs_loaded, 0)
    pG = pG_with_property_csvs_loaded
    selection = '_TYPE_=="transactions"'

    sg = pG.extract_subgraph(
        create_using=cugraph.MultiGraph(directed=True),
        selection=pG.select_edges(selection),
        renumber_graph=False,
    )
    remote_sg = rpG.extract_subgraph(
        create_using="MultiGraph(directed=True)",
        selection=selection,
        renumber_graph=False,
    )

    res_local = call_cugraph_algorithm(
        "uniform_neighbor_sample",
        sg,
        [89021, 89216],
        [10],
        with_replacement=True,
        backend="cudf",
    )
    res_remote = call_cugraph_algorithm(
        "uniform_neighbor_sample",
        remote_sg,
        [89021, 89216],
        [10],
        with_replacement=True,
        backend="cudf",
    )

    assert (res_local["sources"] == res_remote["sources"]).all()
    assert (res_local["destinations"] == res_remote["destinations"]).all()
    assert (res_local["indices"] == res_remote["indices"]).all()


def test_remote_graph_neighbor_sample_implicit_subgraph(
    client_with_property_csvs_loaded, pG_with_property_csvs_loaded
):
    # FIXME: consider moving the call dispatcher into cugraph-service-client or
    # cugraph proper. Import it here for now to allow tests to run in an
    # environment without cugraph-pyg.
    from cugraph_pyg.loader.dispatch import call_cugraph_algorithm

    rpG = RemoteGraph(client_with_property_csvs_loaded, 0)
    pG = pG_with_property_csvs_loaded

    sg = pG.extract_subgraph(
        create_using=cugraph.MultiGraph(directed=True),
        renumber_graph=True,
    )

    res_local = call_cugraph_algorithm(
        "uniform_neighbor_sample",
        sg,
        [89021, 89216],
        [10],
        with_replacement=True,
        backend="cudf",
    )
    res_remote = call_cugraph_algorithm(
        "uniform_neighbor_sample",
        rpG,
        [89021, 89216],
        [10],
        with_replacement=True,
        backend="cudf",
    )

    assert (res_local["sources"] == res_remote["sources"]).all()
    assert (res_local["destinations"] == res_remote["destinations"]).all()
    assert (res_local["indices"] == res_remote["indices"]).all()


@pytest.mark.skip(reason="FIXME: this may fail in CI")
def test_remote_graph_renumber_vertices(
    client_with_property_csvs_loaded, pG_with_property_csvs_loaded
):
    rpG = RemoteGraph(client_with_property_csvs_loaded, 0)
    pG = pG_with_property_csvs_loaded

    re_local = pG.renumber_vertices_by_type()
    re_remote = rpG.renumber_vertices_by_type()

    assert re_local == re_remote

    for k in range(len(re_remote)):
        start = re_remote["start"][k]
        stop = re_remote["stop"][k]
        for i in range(start, stop + 1):
            assert (
                rpG.get_vertex_data(vertex_ids=[i])[rpG.type_col_name][0]
                == re_remote.index[k]
            )


@pytest.mark.skip(reason="FIXME: this may fail in CI")
def test_remote_graph_renumber_edges(
    client_with_property_csvs_loaded, pG_with_property_csvs_loaded
):
    rpG = RemoteGraph(client_with_property_csvs_loaded, 0)
    pG = pG_with_property_csvs_loaded

    re_local = pG.renumber_edges_by_type()
    re_remote = rpG.renumber_edges_by_type()

    assert re_local == re_remote

    for k in range(len(re_remote)):
        start = re_remote["start"][k]
        stop = re_remote["stop"][k]
        for i in range(start, stop + 1):
            assert (
                rpG.get_edge_data(edge_ids=[i])[rpG.type_col_name][0]
                == re_remote.index[k]
            )
