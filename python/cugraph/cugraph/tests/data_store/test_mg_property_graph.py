# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

import dask_cudf
import pytest
import pandas as pd
import cudf
import cupy as cp
import numpy as np
from cudf.testing import assert_frame_equal, assert_series_equal
from cupy.testing import assert_array_equal
from pylibcugraph.testing.utils import gen_fixture_params_product

import cugraph.dask as dcg
from cugraph.experimental.datasets import cyber
from cugraph.experimental.datasets import netscience

# If the rapids-pytest-benchmark plugin is installed, the "gpubenchmark"
# fixture will be available automatically. Check that this fixture is available
# by trying to import rapids_pytest_benchmark, and if that fails, set
# "gpubenchmark" to the standard "benchmark" fixture provided by
# pytest-benchmark.
try:
    import rapids_pytest_benchmark  # noqa: F401
except ImportError:
    import pytest_benchmark

    gpubenchmark = pytest_benchmark.plugin.benchmark

import cugraph


def type_is_categorical(pG):
    return (
        pG._vertex_prop_dataframe is None
        or pG._vertex_prop_dataframe.dtypes[pG.type_col_name] == "category"
    ) and (
        pG._edge_prop_dataframe is None
        or pG._edge_prop_dataframe.dtypes[pG.type_col_name] == "category"
    )


# =============================================================================
# Test data
# =============================================================================

dataset1 = {
    "merchants": [
        [
            "merchant_id",
            "merchant_location",
            "merchant_size",
            "merchant_sales",
            "merchant_num_employees",
            "merchant_name",
        ],
        [
            (11, 78750, 44, 123.2, 12, "north"),
            (4, 78757, 112, 234.99, 18, "south"),
            (21, 44145, 83, 992.1, 27, "east"),
            (16, 47906, 92, 32.43, 5, "west"),
            (86, 47906, 192, 2.43, 51, "west"),
        ],
    ],
    "users": [
        ["user_id", "user_location", "vertical"],
        [
            (89021, 78757, 0),
            (32431, 78750, 1),
            (89216, 78757, 1),
            (78634, 47906, 0),
        ],
    ],
    "transactions": [
        ["user_id", "merchant_id", "volume", "time", "card_num", "card_type"],
        [
            (89021, 11, 33.2, 1639084966.5513437, 123456, "MC"),
            (89216, 4, None, 1639085163.481217, 8832, "CASH"),
            (78634, 16, 72.0, 1639084912.567394, 4321, "DEBIT"),
            (32431, 4, 103.2, 1639084721.354346, 98124, "V"),
        ],
    ],
    "relationships": [
        ["user_id_1", "user_id_2", "relationship_type"],
        [
            (89216, 89021, 9),
            (89216, 32431, 9),
            (32431, 78634, 8),
            (78634, 89216, 8),
        ],
    ],
    "referrals": [
        ["user_id_1", "user_id_2", "merchant_id", "stars"],
        [
            (89216, 78634, 11, 5),
            (89021, 89216, 4, 4),
            (89021, 89216, 21, 3),
            (89021, 89216, 11, 3),
            (89021, 78634, 21, 4),
            (78634, 32431, 11, 4),
        ],
    ],
}


dataset2 = {
    "simple": [
        ["src", "dst", "some_property"],
        [
            (99, 22, "a"),
            (98, 34, "b"),
            (97, 56, "c"),
            (96, 88, "d"),
        ],
    ],
}

# Placeholder for a directed Graph instance. This is not constructed here in
# order to prevent cuGraph code from running on import, which would prevent
# proper pytest collection if an exception is raised. See setup_function().
DiGraph_inst = None


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    global DiGraph_inst

    gc.collect()
    # Set the global DiGraph_inst. This is used for calls that require a Graph
    # type or instance to be provided for tests that use a directed graph.
    DiGraph_inst = cugraph.Graph(directed=True)


# =============================================================================
# Pytest fixtures
# =============================================================================
df_types = [cudf.DataFrame]


def df_type_id(dataframe_type):
    """
    Return a string that describes the dataframe_type, used for test output.
    """
    s = "df_type="
    if dataframe_type == cudf.DataFrame:
        return s + "cudf.DataFrame"
    if dataframe_type == pd.DataFrame:
        return s + "pandas.DataFrame"
    if dataframe_type == dask_cudf.core.DataFrame:
        return s + "dask_cudf.core.DataFrame"
    return s + "?"


df_types_fixture_params = gen_fixture_params_product((df_types, df_type_id))


@pytest.fixture(scope="module", params=df_types_fixture_params)
def net_PropertyGraph(request):
    """
    Fixture which returns an instance of a PropertyGraph with vertex and edge
    data added from the netscience.csv dataset, parameterized for different
    DataFrame types.
    """
    from cugraph.experimental import PropertyGraph

    dataframe_type = request.param[0]
    netscience_csv = netscience.get_path()
    source_col_name = "src"
    dest_col_name = "dst"

    if dataframe_type is pd.DataFrame:
        read_csv = pd.read_csv
    else:
        read_csv = cudf.read_csv
    df = read_csv(
        netscience_csv,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    pG = PropertyGraph()
    pG.add_edge_data(df, (source_col_name, dest_col_name))

    return pG


@pytest.fixture(scope="module", params=df_types_fixture_params)
def dataset1_PropertyGraph(request):
    """
    Fixture which returns an instance of a PropertyGraph with vertex and edge
    data added from dataset1, parameterized for different DataFrame types.
    """
    dataframe_type = request.param[0]
    from cugraph.experimental import PropertyGraph

    (merchants, users, transactions, relationships, referrals) = dataset1.values()

    pG = PropertyGraph()

    # Vertex and edge data is added as one or more DataFrames; either a Pandas
    # DataFrame to keep data on the CPU, a cuDF DataFrame to keep data on GPU,
    # or a dask_cudf DataFrame to keep data on distributed GPUs.

    # For dataset1: vertices are merchants and users, edges are transactions,
    # relationships, and referrals.

    # property_columns=None (the default) means all columns except
    # vertex_col_name will be used as properties for the vertices/edges.

    pG.add_vertex_data(
        dataframe_type(columns=merchants[0], data=merchants[1]),
        type_name="merchants",
        vertex_col_name="merchant_id",
        property_columns=None,
    )
    pG.add_vertex_data(
        dataframe_type(columns=users[0], data=users[1]),
        type_name="users",
        vertex_col_name="user_id",
        property_columns=None,
    )

    pG.add_edge_data(
        dataframe_type(columns=transactions[0], data=transactions[1]),
        type_name="transactions",
        vertex_col_names=("user_id", "merchant_id"),
        property_columns=None,
    )
    pG.add_edge_data(
        dataframe_type(columns=relationships[0], data=relationships[1]),
        type_name="relationships",
        vertex_col_names=("user_id_1", "user_id_2"),
        property_columns=None,
    )
    pG.add_edge_data(
        dataframe_type(columns=referrals[0], data=referrals[1]),
        type_name="referrals",
        vertex_col_names=("user_id_1", "user_id_2"),
        property_columns=None,
    )
    assert type_is_categorical(pG)
    return (pG, dataset1)


@pytest.fixture(scope="function")
def dataset1_MGPropertyGraph(dask_client):
    """
    Fixture which returns an instance of a PropertyGraph with vertex and edge
    data added from dataset1, parameterized for different DataFrame types.
    """
    dataframe_type = cudf.DataFrame
    (merchants, users, transactions, relationships, referrals) = dataset1.values()
    from cugraph.experimental import MGPropertyGraph

    mpG = MGPropertyGraph()

    # Vertex and edge data is added as one or more DataFrames; either a Pandas
    # DataFrame to keep data on the CPU, a cuDF DataFrame to keep data on GPU,
    # or a dask_cudf DataFrame to keep data on distributed GPUs.

    # For dataset1: vertices are merchants and users, edges are transactions,
    # relationships, and referrals.

    # property_columns=None (the default) means all columns except
    # vertex_col_name will be used as properties for the vertices/edges.

    sg_df = dataframe_type(columns=merchants[0], data=merchants[1])
    mg_df = dask_cudf.from_cudf(sg_df, npartitions=2)
    mpG.add_vertex_data(
        mg_df,
        type_name="merchants",
        vertex_col_name="merchant_id",
        property_columns=None,
    )

    sg_df = dataframe_type(columns=users[0], data=users[1])
    mg_df = dask_cudf.from_cudf(sg_df, npartitions=2)
    mpG.add_vertex_data(
        mg_df, type_name="users", vertex_col_name="user_id", property_columns=None
    )

    sg_df = dataframe_type(columns=transactions[0], data=transactions[1])
    mg_df = dask_cudf.from_cudf(sg_df, npartitions=2)
    mpG.add_edge_data(
        mg_df,
        type_name="transactions",
        vertex_col_names=("user_id", "merchant_id"),
        property_columns=None,
    )

    sg_df = dataframe_type(columns=relationships[0], data=relationships[1])
    mg_df = dask_cudf.from_cudf(sg_df, npartitions=2)
    mpG.add_edge_data(
        mg_df,
        type_name="relationships",
        vertex_col_names=("user_id_1", "user_id_2"),
        property_columns=None,
    )

    sg_df = dataframe_type(columns=referrals[0], data=referrals[1])
    mg_df = dask_cudf.from_cudf(sg_df, npartitions=2)
    mpG.add_edge_data(
        mg_df,
        type_name="referrals",
        vertex_col_names=("user_id_1", "user_id_2"),
        property_columns=None,
    )

    assert type_is_categorical(mpG)
    return (mpG, dataset1)


@pytest.fixture(scope="module")
def dataset2_simple_MGPropertyGraph(dask_client):
    from cugraph.experimental import MGPropertyGraph

    dataframe_type = cudf.DataFrame
    simple = dataset2["simple"]
    mpG = MGPropertyGraph()

    sg_df = dataframe_type(columns=simple[0], data=simple[1])
    mgdf = dask_cudf.from_cudf(sg_df, npartitions=2)

    mpG.add_edge_data(mgdf, vertex_col_names=("src", "dst"))

    assert type_is_categorical(mpG)
    return (mpG, simple)


@pytest.fixture(scope="module")
def dataset2_MGPropertyGraph(dask_client):
    from cugraph.experimental import MGPropertyGraph

    dataframe_type = cudf.DataFrame
    simple = dataset2["simple"]
    mpG = MGPropertyGraph()

    sg_df = dataframe_type(columns=simple[0], data=simple[1])
    mgdf = dask_cudf.from_cudf(sg_df, npartitions=2)

    mpG.add_edge_data(mgdf, vertex_col_names=("src", "dst"))

    assert type_is_categorical(mpG)
    return (mpG, simple)


@pytest.fixture(scope="module", params=df_types_fixture_params)
def net_MGPropertyGraph(dask_client):
    """
    Fixture which returns an instance of a PropertyGraph with vertex and edge
    data added from the netscience.csv dataset, parameterized for different
    DataFrame types.
    """
    from cugraph.experimental import MGPropertyGraph

    input_data_path = str(netscience.get_path())
    print(f"dataset={input_data_path}")
    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    dpG = MGPropertyGraph()
    dpG.add_edge_data(ddf, ("src", "dst"))
    assert type_is_categorical(dpG)
    return dpG


@pytest.mark.skip(reason="Skipping tests because it is a work in progress")
def test_extract_subgraph_no_query(net_MGPropertyGraph, net_PropertyGraph):
    """
    Call extract with no args, should result in the entire property graph.
    """
    dpG = net_MGPropertyGraph
    pG = net_PropertyGraph
    assert pG.get_num_edges() == dpG.get_num_edges()
    assert pG.get_num_vertices() == dpG.get_num_vertices()
    # tests that the edges are the same in the sg and mg property graph
    sg_df = pG.edges.sort_values(by=["_SRC_", "_DST_"]).reset_index(drop=True)
    mg_df = dpG.edges.compute().sort_values(by=["_SRC_", "_DST_"])
    mg_df = mg_df.reset_index(drop=True)
    assert sg_df.equals(mg_df)
    subgraph = pG.extract_subgraph()
    dask_subgraph = dpG.extract_subgraph()
    sg_subgraph_df = subgraph.edge_data.sort_values(by=list(subgraph.edge_data.columns))
    sg_subgraph_df = sg_subgraph_df.reset_index(drop=True)
    mg_subgraph_df = dask_subgraph.edge_data.compute()
    mg_subgraph_df = mg_subgraph_df.sort_values(by=list(mg_subgraph_df.columns))
    mg_subgraph_df = mg_subgraph_df.reset_index(drop=True)
    assert sg_subgraph_df[["_SRC_", "_DST_"]].equals(mg_subgraph_df[["_SRC_", "_DST_"]])
    assert sg_subgraph_df.dtypes["_TYPE_"] == "category"
    assert mg_subgraph_df.dtypes["_TYPE_"] == "category"


@pytest.mark.skip(reason="Skipping tests because it is a work in progress")
def test_adding_fixture(dataset1_PropertyGraph, dataset1_MGPropertyGraph):
    (sgpG, _) = dataset1_PropertyGraph
    (mgPG, _) = dataset1_MGPropertyGraph
    subgraph = sgpG.extract_subgraph()
    dask_subgraph = mgPG.extract_subgraph()
    sg_subgraph_df = subgraph.edge_data.sort_values(by=list(subgraph.edge_data.columns))
    sg_subgraph_df = sg_subgraph_df.reset_index(drop=True)
    mg_subgraph_df = dask_subgraph.edge_data.compute()
    mg_subgraph_df = mg_subgraph_df.sort_values(by=list(mg_subgraph_df.columns))
    mg_subgraph_df = mg_subgraph_df.reset_index(drop=True)
    assert sg_subgraph_df[["_SRC_", "_DST_"]].equals(mg_subgraph_df[["_SRC_", "_DST_"]])
    assert sg_subgraph_df.dtypes["_TYPE_"] == "category"
    assert mg_subgraph_df.dtypes["_TYPE_"] == "category"


@pytest.mark.skip(reason="Skipping tests because it is a work in progress")
def test_frame_data(dataset1_PropertyGraph, dataset1_MGPropertyGraph):
    (sgpG, _) = dataset1_PropertyGraph
    (mgpG, _) = dataset1_MGPropertyGraph

    edge_sort_col = ["_SRC_", "_DST_", "_TYPE_"]
    vert_sort_col = ["_VERTEX_", "_TYPE_"]
    # vertex_prop_dataframe
    sg_vp_df = sgpG._vertex_prop_dataframe.sort_values(by=vert_sort_col).reset_index(
        drop=True
    )
    mg_vp_df = (
        mgpG._vertex_prop_dataframe.compute()
        .sort_values(by=vert_sort_col)
        .reset_index(drop=True)
    )
    assert sg_vp_df["_VERTEX_"].equals(mg_vp_df["_VERTEX_"])

    # get_edge_prop_dataframe
    sg_ep_df = sgpG._edge_prop_dataframe.sort_values(by=edge_sort_col).reset_index(
        drop=True
    )
    mg_ep_df = (
        mgpG._edge_prop_dataframe.compute()
        .sort_values(by=edge_sort_col)
        .reset_index(drop=True)
    )
    assert sg_ep_df["_SRC_"].equals(mg_ep_df["_SRC_"])
    assert sg_ep_df.dtypes["_TYPE_"] == "category"
    assert mg_ep_df.dtypes["_TYPE_"] == "category"


def test_add_edge_data_with_ids(dask_client):
    """
    add_edge_data() on "transactions" table, all properties.
    """
    from cugraph.experimental import MGPropertyGraph

    transactions = dataset1["transactions"]
    transactions_df = cudf.DataFrame(columns=transactions[0], data=transactions[1])
    transactions_df["edge_id"] = list(range(10, 10 + len(transactions_df)))
    transactions_df = dask_cudf.from_cudf(transactions_df, npartitions=2)

    pG = MGPropertyGraph()
    pG.add_edge_data(
        transactions_df,
        type_name="transactions",
        edge_id_col_name="edge_id",
        vertex_col_names=("user_id", "merchant_id"),
        property_columns=None,
    )

    assert pG.get_num_vertices() == 7
    # 'transactions' is edge type, not vertex type
    assert pG.get_num_vertices("transactions") == 0
    assert pG.get_num_edges() == 4
    assert pG.get_num_edges("transactions") == 4
    # Original SRC and DST columns no longer include "merchant_id", "user_id"
    expected_props = ["volume", "time", "card_num", "card_type"]
    assert sorted(pG.edge_property_names) == sorted(expected_props)

    relationships = dataset1["relationships"]
    relationships_df = cudf.DataFrame(columns=relationships[0], data=relationships[1])

    # user-provided, then auto-gen (not allowed)
    with pytest.raises(NotImplementedError):
        pG.add_edge_data(
            dask_cudf.from_cudf(relationships_df, npartitions=2),
            type_name="relationships",
            vertex_col_names=("user_id_1", "user_id_2"),
            property_columns=None,
        )

    relationships_df["edge_id"] = list(range(30, 30 + len(relationships_df)))
    relationships_df = dask_cudf.from_cudf(relationships_df, npartitions=2)

    pG.add_edge_data(
        relationships_df,
        type_name="relationships",
        edge_id_col_name="edge_id",
        vertex_col_names=("user_id_1", "user_id_2"),
        property_columns=None,
    )

    df = pG.get_edge_data(types="transactions").compute()
    assert_series_equal(
        df[pG.edge_id_col_name].sort_values().reset_index(drop=True),
        transactions_df["edge_id"].compute(),
        check_names=False,
    )
    df = pG.get_edge_data(types="relationships").compute()
    assert_series_equal(
        df[pG.edge_id_col_name].sort_values().reset_index(drop=True),
        relationships_df["edge_id"].compute(),
        check_names=False,
    )

    # auto-gen, then user-provided (not allowed)
    pG = MGPropertyGraph()
    pG.add_edge_data(
        transactions_df,
        type_name="transactions",
        vertex_col_names=("user_id", "merchant_id"),
        property_columns=None,
    )
    with pytest.raises(NotImplementedError):
        pG.add_edge_data(
            relationships_df,
            type_name="relationships",
            edge_id_col_name="edge_id",
            vertex_col_names=("user_id_1", "user_id_2"),
            property_columns=None,
        )


def test_property_names_attrs(dataset1_MGPropertyGraph):
    """
    Ensure the correct number of user-visible properties for vertices and edges
    are returned. This should exclude the internal bookkeeping properties.
    """
    (pG, data) = dataset1_MGPropertyGraph

    # _VERTEX_ columns: "merchant_id", "user_id"
    expected_vert_prop_names = [
        "merchant_location",
        "merchant_size",
        "merchant_sales",
        "merchant_num_employees",
        "user_location",
        "merchant_name",
        "vertical",
    ]
    # _SRC_ and _DST_ columns: "user_id", "user_id_1", "user_id_2"
    # Note that "merchant_id" is a property in for type "transactions"
    expected_edge_prop_names = [
        "merchant_id",
        "volume",
        "time",
        "card_num",
        "card_type",
        "relationship_type",
        "stars",
    ]

    # Extracting a subgraph with weights has/had a side-effect of adding a
    # weight column, so call extract_subgraph() to ensure the internal weight
    # column name is not present.
    pG.extract_subgraph(default_edge_weight=1.0)

    actual_vert_prop_names = pG.vertex_property_names
    actual_edge_prop_names = pG.edge_property_names

    assert sorted(actual_vert_prop_names) == sorted(expected_vert_prop_names)
    assert sorted(actual_edge_prop_names) == sorted(expected_edge_prop_names)


def test_extract_subgraph_nonrenumbered_noedgedata(dataset2_simple_MGPropertyGraph):
    """
    Ensure a subgraph can be extracted that contains no edge_data.  Also ensure
    renumber cannot be False since that is currently not allowed for MG.
    """
    from cugraph import Graph

    (pG, data) = dataset2_simple_MGPropertyGraph

    # renumber=False is currently not allowed for MG.
    with pytest.raises(ValueError):
        G = pG.extract_subgraph(
            create_using=Graph(directed=True), renumber_graph=False, add_edge_data=False
        )

    G = pG.extract_subgraph(create_using=Graph(directed=True), add_edge_data=False)

    actual_edgelist = G.edgelist.edgelist_df.compute()

    src_col_name = pG.src_col_name
    dst_col_name = pG.dst_col_name

    # create a DF without the properties (ie. the last column)
    expected_edgelist = cudf.DataFrame(
        columns=[src_col_name, dst_col_name], data=[(i, j) for (i, j, k) in data[1]]
    )

    assert_frame_equal(
        expected_edgelist.sort_values(by=src_col_name, ignore_index=True),
        actual_edgelist.sort_values(by=src_col_name, ignore_index=True),
    )
    assert hasattr(G, "edge_data") is False


def test_num_vertices_with_properties(dataset2_simple_MGPropertyGraph):
    """
    Checks that the num_vertices_with_properties attr is set to the number of
    vertices that have properties, as opposed to just num_vertices which also
    includes all verts in the graph edgelist.
    """
    (pG, data) = dataset2_simple_MGPropertyGraph

    # assume no repeated vertices
    assert pG.get_num_vertices() == len(data[1]) * 2
    assert pG.get_num_vertices(include_edge_data=False) == 0

    df = cudf.DataFrame(
        {
            "vertex": [98, 97],
            "some_property": ["a", "b"],
        }
    )
    mgdf = dask_cudf.from_cudf(df, npartitions=2)
    pG.add_vertex_data(mgdf, vertex_col_name="vertex")

    # assume no repeated vertices
    assert pG.get_num_vertices() == len(data[1]) * 2
    assert pG.get_num_vertices(include_edge_data=False) == 2
    assert type_is_categorical(pG)


def test_edges_attr(dataset2_simple_MGPropertyGraph):
    """
    Ensure the edges attr returns the src, dst, edge_id columns properly.
    """
    (pG, data) = dataset2_simple_MGPropertyGraph

    # create a DF without the properties (ie. the last column)
    expected_edges = cudf.DataFrame(
        columns=[pG.src_col_name, pG.dst_col_name],
        data=[(i, j) for (i, j, k) in data[1]],
    )
    actual_edges = pG.edges[[pG.src_col_name, pG.dst_col_name]].compute()

    assert_frame_equal(
        expected_edges.sort_values(by=pG.src_col_name, ignore_index=True),
        actual_edges.sort_values(by=pG.src_col_name, ignore_index=True),
    )
    edge_ids = pG.edges[pG.edge_id_col_name].compute()
    expected_num_edges = len(data[1])
    assert len(edge_ids) == expected_num_edges
    assert edge_ids.nunique() == expected_num_edges


def test_get_vertex_data(dataset1_MGPropertyGraph):
    """
    Ensure PG.get_vertex_data() returns the correct data based on vertex IDs
    passed in.
    """
    (pG, data) = dataset1_MGPropertyGraph

    # Ensure the generated vertex IDs are unique
    all_vertex_data = pG.get_vertex_data()
    assert all_vertex_data[pG.vertex_col_name].nunique().compute() == len(
        all_vertex_data
    )

    # Test with specific columns and types
    vert_type = "merchants"
    columns = ["merchant_location", "merchant_size"]

    some_vertex_data = pG.get_vertex_data(types=[vert_type], columns=columns)
    # Ensure the returned df is the right length and includes only the
    # vert/type + specified columns
    standard_vert_columns = [pG.vertex_col_name, pG.type_col_name]
    assert len(some_vertex_data) == len(data[vert_type][1])
    assert sorted(some_vertex_data.columns) == sorted(columns + standard_vert_columns)
    assert some_vertex_data.dtypes["_TYPE_"] == "category"

    # Test with all params specified
    vert_ids = [11, 4, 21]
    vert_type = "merchants"
    columns = ["merchant_location", "merchant_size"]

    some_vertex_data = pG.get_vertex_data(
        vertex_ids=vert_ids, types=[vert_type], columns=columns
    )
    # Ensure the returned df is the right length and includes at least the
    # specified columns.
    assert len(some_vertex_data) == len(vert_ids)
    assert set(columns) - set(some_vertex_data.columns) == set()
    assert some_vertex_data.dtypes["_TYPE_"] == "category"

    # Allow a single vertex type and single vertex id to be passed in
    df1 = pG.get_vertex_data(vertex_ids=[11], types=[vert_type]).compute()
    df2 = pG.get_vertex_data(vertex_ids=11, types=vert_type).compute()
    assert len(df1) == 1
    assert df1.shape == df2.shape
    assert_frame_equal(df1, df2, check_like=True)


def test_get_vertex_data_repeated(dask_client):
    from cugraph.experimental import MGPropertyGraph

    df = cudf.DataFrame({"vertex": [2, 3, 4, 1], "feat": [0, 1, 2, 3]})
    df = dask_cudf.from_cudf(df, npartitions=2)
    pG = MGPropertyGraph()
    pG.add_vertex_data(df, "vertex")
    df1 = pG.get_vertex_data(vertex_ids=[2, 1, 3, 1], columns=["feat"])
    df1 = df1.compute()
    expected = cudf.DataFrame(
        {
            pG.vertex_col_name: [2, 1, 3, 1],
            pG.type_col_name: ["", "", "", ""],
            "feat": [0, 3, 1, 3],
        }
    )
    df1[pG.type_col_name] = df1[pG.type_col_name].astype(str)  # Undo category
    assert_frame_equal(df1, expected)


def test_get_edge_data(dataset1_MGPropertyGraph):
    """
    Ensure PG.get_edge_data() returns the correct data based on edge IDs passed
    in.
    """
    (pG, data) = dataset1_MGPropertyGraph

    # Ensure the generated edge IDs are unique
    all_edge_data = pG.get_edge_data()
    assert all_edge_data[pG.edge_id_col_name].nunique().compute() == len(all_edge_data)

    # Test with specific edge IDs
    edge_ids = [4, 5, 6]
    some_edge_data = pG.get_edge_data(edge_ids)
    actual_edge_ids = some_edge_data[pG.edge_id_col_name].compute()
    if hasattr(actual_edge_ids, "values_host"):
        actual_edge_ids = actual_edge_ids.values_host
    assert sorted(actual_edge_ids) == sorted(edge_ids)
    assert some_edge_data.dtypes["_TYPE_"] == "category"

    # Create a list of expected column names from the three input tables
    expected_columns = set(
        [pG.src_col_name, pG.dst_col_name, pG.edge_id_col_name, pG.type_col_name]
    )
    for d in ["transactions", "relationships", "referrals"]:
        for name in data[d][0]:
            expected_columns.add(name)
    expected_columns -= {"user_id", "user_id_1", "user_id_2"}

    actual_columns = set(some_edge_data.columns)

    assert actual_columns == expected_columns

    # Test with specific columns and types
    edge_type = "transactions"
    columns = ["card_num", "card_type"]

    some_edge_data = pG.get_edge_data(types=[edge_type], columns=columns)
    # Ensure the returned df is the right length and includes only the
    # src/dst/id/type + specified columns
    standard_edge_columns = [
        pG.src_col_name,
        pG.dst_col_name,
        pG.edge_id_col_name,
        pG.type_col_name,
    ]
    assert len(some_edge_data) == len(data[edge_type][1])
    assert sorted(some_edge_data.columns) == sorted(columns + standard_edge_columns)
    assert some_edge_data.dtypes["_TYPE_"] == "category"

    # Test with all params specified
    # FIXME: since edge IDs are generated, assume that these are correct based
    # on the intended edges being the first three added.
    edge_ids = [0, 1, 2]
    edge_type = "transactions"
    columns = ["card_num", "card_type"]
    some_edge_data = pG.get_edge_data(
        edge_ids=edge_ids, types=[edge_type], columns=columns
    )
    # Ensure the returned df is the right length and includes at least the
    # specified columns.
    assert len(some_edge_data) == len(edge_ids)
    assert set(columns) - set(some_edge_data.columns) == set()
    assert some_edge_data.dtypes["_TYPE_"] == "category"

    # Allow a single edge type and single edge id to be passed in
    df1 = pG.get_edge_data(edge_ids=[1], types=[edge_type]).compute()
    df2 = pG.get_edge_data(edge_ids=1, types=edge_type).compute()
    assert len(df1) == 1
    assert df1.shape == df2.shape
    assert_frame_equal(df1, df2, check_like=True)


def test_get_edge_data_repeated(dask_client):
    from cugraph.experimental import MGPropertyGraph

    df = cudf.DataFrame(
        {"src": [1, 1, 1, 2], "dst": [2, 3, 4, 1], "edge_feat": [0, 1, 2, 3]}
    )
    df = dask_cudf.from_cudf(df, npartitions=2)
    pG = MGPropertyGraph()
    pG.add_edge_data(df, vertex_col_names=["src", "dst"])
    df1 = pG.get_edge_data(edge_ids=[2, 1, 3, 1], columns=["edge_feat"])
    df1 = df1.compute()
    expected = cudf.DataFrame(
        {
            pG.edge_id_col_name: [2, 1, 3, 1],
            pG.src_col_name: [1, 1, 2, 1],
            pG.dst_col_name: [4, 3, 1, 3],
            pG.type_col_name: ["", "", "", ""],
            "edge_feat": [2, 1, 3, 1],
        }
    )
    df1[pG.type_col_name] = df1[pG.type_col_name].astype(str)  # Undo category

    # Order and indices don't matter
    df1 = df1.sort_values(df1.columns).reset_index(drop=True)
    expected = expected.sort_values(df1.columns).reset_index(drop=True)
    assert_frame_equal(df1, expected)


def test_get_data_empty_graphs(dask_client):
    """
    Ensures that calls to pG.get_*_data() on an empty pG are handled correctly.
    """
    from cugraph.experimental import MGPropertyGraph

    pG = MGPropertyGraph()

    assert pG.get_vertex_data() is None
    assert pG.get_vertex_data([0, 1, 2]) is None
    assert pG.get_edge_data() is None
    assert pG.get_edge_data([0, 1, 2]) is None


@pytest.mark.parametrize("prev_id_column", [None, "prev_id"])
def test_renumber_vertices_by_type(dataset1_MGPropertyGraph, prev_id_column):
    from cugraph.experimental import MGPropertyGraph

    (pG, data) = dataset1_MGPropertyGraph
    with pytest.raises(ValueError, match="existing column"):
        pG.renumber_vertices_by_type("merchant_size")
    vertex_property_names = set(pG.vertex_property_names)
    edge_property_names = set(pG.edge_property_names)
    df_id_ranges = pG.renumber_vertices_by_type(prev_id_column)
    if prev_id_column is not None:
        vertex_property_names.add(prev_id_column)
    assert vertex_property_names == set(pG.vertex_property_names)
    assert edge_property_names == set(pG.edge_property_names)
    expected = {
        "merchants": [0, 4],  # stop is inclusive
        "users": [5, 8],
    }
    for key, (start, stop) in expected.items():
        assert df_id_ranges.loc[key, "start"] == start
        assert df_id_ranges.loc[key, "stop"] == stop
        df = pG.get_vertex_data(types=[key]).compute().to_pandas()
        df = df.reset_index(drop=True)
        assert len(df) == stop - start + 1
        assert (df["_VERTEX_"] == pd.Series(range(start, stop + 1))).all()
        if prev_id_column is not None:
            cur = df[prev_id_column].sort_values()
            expected = sorted(x for x, *args in data[key][1])
            assert (cur == pd.Series(expected, index=cur.index)).all()

    # Make sure we renumber vertex IDs in edge data too
    df = pG.get_edge_data().compute()
    assert 0 <= df[pG.src_col_name].min() < df[pG.src_col_name].max() < 9
    assert 0 <= df[pG.dst_col_name].min() < df[pG.dst_col_name].max() < 9

    empty_pG = MGPropertyGraph()
    assert empty_pG.renumber_vertices_by_type(prev_id_column) is None

    # Test when vertex IDs only exist in edge data
    df = cudf.DataFrame({"src": [99998], "dst": [99999]})
    df = dask_cudf.from_cudf(df, npartitions=1)
    empty_pG.add_edge_data(df, ["src", "dst"])
    with pytest.raises(NotImplementedError, match="only exist in edge"):
        empty_pG.renumber_vertices_by_type(prev_id_column)


@pytest.mark.parametrize("prev_id_column", [None, "prev_id"])
def test_renumber_edges_by_type(dataset1_MGPropertyGraph, prev_id_column):
    from cugraph.experimental import MGPropertyGraph

    (pG, data) = dataset1_MGPropertyGraph
    with pytest.raises(ValueError, match="existing column"):
        pG.renumber_edges_by_type("time")
    df_id_ranges = pG.renumber_edges_by_type(prev_id_column)
    expected = {
        "referrals": [0, 5],  # stop is inclusive
        "relationships": [6, 9],
        "transactions": [10, 13],
    }
    for key, (start, stop) in expected.items():
        assert df_id_ranges.loc[key, "start"] == start
        assert df_id_ranges.loc[key, "stop"] == stop
        df = pG.get_edge_data(types=[key]).compute().to_pandas()
        df = df.reset_index(drop=True)
        assert len(df) == stop - start + 1
        assert (df[pG.edge_id_col_name] == pd.Series(range(start, stop + 1))).all()

        if prev_id_column is not None:
            assert prev_id_column in df.columns

    empty_pG = MGPropertyGraph()
    assert empty_pG.renumber_edges_by_type(prev_id_column) is None


def test_renumber_vertices_edges_dtypes(dask_client):
    from cugraph.experimental import MGPropertyGraph

    edgelist_df = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "src": cp.array([0, 5, 2, 3, 4, 3], dtype="int32"),
                "dst": cp.array([2, 4, 4, 5, 1, 2], dtype="int32"),
                "eid": cp.array([8, 7, 5, 2, 9, 1], dtype="int32"),
            }
        ),
        npartitions=2,
    )

    vertex_df = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "v": cp.array([0, 1, 2, 3, 4, 5], dtype="int32"),
                "p": [5, 10, 15, 20, 25, 30],
            }
        ),
        npartitions=2,
    )

    pG = MGPropertyGraph()
    pG.add_vertex_data(
        vertex_df, vertex_col_name="v", property_columns=["p"], type_name="vt1"
    )
    pG.add_edge_data(
        edgelist_df,
        vertex_col_names=["src", "dst"],
        edge_id_col_name="eid",
        type_name="et1",
    )

    pG.renumber_vertices_by_type()
    vd = pG.get_vertex_data()
    assert vd.index.dtype == cp.int32

    pG.renumber_edges_by_type()
    ed = pG.get_edge_data()
    assert ed[pG.edge_id_col_name].dtype == cp.int32


def test_add_data_noncontiguous(dask_client):
    from cugraph.experimental import MGPropertyGraph

    df = cudf.DataFrame(
        {
            "src": [0, 0, 1, 2, 2, 3, 3, 1, 2, 4],
            "dst": [1, 2, 4, 3, 3, 1, 2, 4, 4, 3],
            "edge_type": [
                "pig",
                "dog",
                "cat",
                "pig",
                "cat",
                "pig",
                "dog",
                "pig",
                "cat",
                "dog",
            ],
        }
    )
    counts = df["edge_type"].value_counts()
    df = dask_cudf.from_cudf(df, npartitions=2)

    pG = MGPropertyGraph()
    for edge_type in ["cat", "dog", "pig"]:
        pG.add_edge_data(
            df[df.edge_type == edge_type],
            vertex_col_names=["src", "dst"],
            type_name=edge_type,
        )
    for edge_type in ["cat", "dog", "pig"]:
        cur_df = pG.get_edge_data(types=edge_type).compute()
        assert len(cur_df) == counts[edge_type]
        assert_series_equal(
            cur_df[pG.type_col_name].astype(str),
            cur_df["edge_type"],
            check_names=False,
        )

    df["vertex"] = (
        100 * df["src"]
        + df["dst"]
        + df["edge_type"].map({"pig": 0, "dog": 10, "cat": 20})
    )
    pG = MGPropertyGraph()
    for edge_type in ["cat", "dog", "pig"]:
        pG.add_vertex_data(
            df[df.edge_type == edge_type], vertex_col_name="vertex", type_name=edge_type
        )
    for edge_type in ["cat", "dog", "pig"]:
        cur_df = pG.get_vertex_data(types=edge_type).compute()
        assert len(cur_df) == counts[edge_type]
        assert_series_equal(
            cur_df[pG.type_col_name].astype(str),
            cur_df["edge_type"],
            check_names=False,
        )


def test_vertex_vector_property(dask_client):
    from cugraph.experimental import MGPropertyGraph

    (merchants, users, transactions, relationships, referrals) = dataset1.values()

    pG = MGPropertyGraph()
    m_df = cudf.DataFrame(columns=merchants[0], data=merchants[1])
    merchants_df = dask_cudf.from_cudf(m_df, npartitions=2)
    with pytest.raises(ValueError):
        # Column doesn't exist
        pG.add_vertex_data(
            merchants_df,
            type_name="merchants",
            vertex_col_name="merchant_id",
            vector_properties={"vec1": ["merchant_location", "BAD_NAME"]},
        )
    with pytest.raises(ValueError):
        # Using reserved name
        pG.add_vertex_data(
            merchants_df,
            type_name="merchants",
            vertex_col_name="merchant_id",
            vector_properties={
                pG.type_col_name: ["merchant_location", "merchant_size"]
            },
        )
    with pytest.raises(TypeError):
        # String value invalid
        pG.add_vertex_data(
            merchants_df,
            type_name="merchants",
            vertex_col_name="merchant_id",
            vector_properties={"vec1": "merchant_location"},
        )
    with pytest.raises(ValueError):
        # Length-0 vector not allowed
        pG.add_vertex_data(
            merchants_df,
            type_name="merchants",
            vertex_col_name="merchant_id",
            vector_properties={"vec1": []},
        )
    pG.add_vertex_data(
        merchants_df,
        type_name="merchants",
        vertex_col_name="merchant_id",
        vector_properties={
            "vec1": ["merchant_location", "merchant_size", "merchant_num_employees"]
        },
    )
    df = pG.get_vertex_data()
    expected_columns = {
        pG.vertex_col_name,
        pG.type_col_name,
        "merchant_sales",
        "merchant_name",
        "vec1",
    }
    assert set(df.columns) == expected_columns
    expected = m_df[
        ["merchant_location", "merchant_size", "merchant_num_employees"]
    ].values
    expected = expected[np.lexsort(expected.T)]  # may be jumbled, so sort

    vec1 = pG.vertex_vector_property_to_array(df, "vec1").compute()
    vec1 = vec1[np.lexsort(vec1.T)]  # may be jumbled, so sort
    assert_array_equal(expected, vec1)
    vec1 = pG.vertex_vector_property_to_array(df, "vec1", missing="error").compute()
    vec1 = vec1[np.lexsort(vec1.T)]  # may be jumbled, so sort
    assert_array_equal(expected, vec1)
    with pytest.raises(ValueError):
        pG.vertex_vector_property_to_array(df, "BAD_NAME")

    u_df = cudf.DataFrame(columns=users[0], data=users[1])
    users_df = dask_cudf.from_cudf(u_df, npartitions=2)
    with pytest.raises(ValueError):
        # Length doesn't match existing vector
        pG.add_vertex_data(
            users_df,
            type_name="users",
            vertex_col_name="user_id",
            property_columns=["vertical"],
            vector_properties={"vec1": ["user_location", "vertical"]},
        )
    with pytest.raises(ValueError):
        # Can't assign property to existing vector column
        pG.add_vertex_data(
            users_df.assign(vec1=users_df["user_id"]),
            type_name="users",
            vertex_col_name="user_id",
            property_columns=["vec1"],
        )

    pG.add_vertex_data(
        users_df,
        type_name="users",
        vertex_col_name="user_id",
        property_columns=["vertical"],
        vector_properties={"vec2": ["user_location", "vertical"]},
    )
    expected_columns.update({"vec2", "vertical"})
    df = pG.get_vertex_data()
    assert set(df.columns) == expected_columns
    vec1 = pG.vertex_vector_property_to_array(df, "vec1").compute()
    vec1 = vec1[np.lexsort(vec1.T)]  # may be jumbled, so sort
    assert_array_equal(expected, vec1)
    with pytest.raises(RuntimeError):
        pG.vertex_vector_property_to_array(df, "vec1", missing="error").compute()

    pGusers = MGPropertyGraph()
    pGusers.add_vertex_data(
        users_df,
        type_name="users",
        vertex_col_name="user_id",
        vector_property="vec3",
    )
    vec2 = pG.vertex_vector_property_to_array(df, "vec2").compute()
    vec2 = vec2[np.lexsort(vec2.T)]  # may be jumbled, so sort
    df2 = pGusers.get_vertex_data()
    assert set(df2.columns) == {pG.vertex_col_name, pG.type_col_name, "vec3"}
    vec3 = pGusers.vertex_vector_property_to_array(df2, "vec3").compute()
    vec3 = vec3[np.lexsort(vec3.T)]  # may be jumbled, so sort
    assert_array_equal(vec2, vec3)

    vec1filled = pG.vertex_vector_property_to_array(
        df, "vec1", 0, missing="error"
    ).compute()
    vec1filled = vec1filled[np.lexsort(vec1filled.T)]  # may be jumbled, so sort
    expectedfilled = np.concatenate([cp.zeros((4, 3), int), expected])
    assert_array_equal(expectedfilled, vec1filled)

    vec1filled = pG.vertex_vector_property_to_array(df, "vec1", [0, 0, 0]).compute()
    vec1filled = vec1filled[np.lexsort(vec1filled.T)]  # may be jumbled, so sort
    assert_array_equal(expectedfilled, vec1filled)

    with pytest.raises(ValueError, match="expected 3"):
        pG.vertex_vector_property_to_array(df, "vec1", [0, 0]).compute()

    vec2 = pG.vertex_vector_property_to_array(df, "vec2").compute()
    vec2 = vec2[np.lexsort(vec2.T)]  # may be jumbled, so sort
    expected = u_df[["user_location", "vertical"]].values
    expected = expected[np.lexsort(expected.T)]  # may be jumbled, so sort
    assert_array_equal(expected, vec2)
    with pytest.raises(TypeError):
        # Column is wrong type to be a vector
        pG.vertex_vector_property_to_array(
            df.rename(columns={"vec1": "vertical", "vertical": "vec1"}), "vec1"
        )
    with pytest.raises(ValueError):
        # Vector column doesn't exist in dataframe
        pG.vertex_vector_property_to_array(df.rename(columns={"vec1": "moved"}), "vec1")
    with pytest.raises(TypeError):
        # Bad type
        pG.vertex_vector_property_to_array(42, "vec1")


def test_edge_vector_property(dask_client):
    from cugraph.experimental import MGPropertyGraph

    df1 = cudf.DataFrame(
        {
            "src": [0, 1],
            "dst": [1, 2],
            "feat_0": [1, 2],
            "feat_1": [10, 20],
            "feat_2": [10, 20],
        }
    )
    dd1 = dask_cudf.from_cudf(df1, npartitions=2)
    df2 = cudf.DataFrame(
        {
            "src": [2, 3],
            "dst": [1, 2],
            "feat_0": [0.5, 0.2],
            "feat_1": [1.5, 1.2],
        }
    )
    dd2 = dask_cudf.from_cudf(df2, npartitions=2)
    pG = MGPropertyGraph()
    pG.add_edge_data(
        dd1, ("src", "dst"), vector_properties={"vec1": ["feat_0", "feat_1", "feat_2"]}
    )
    df = pG.get_edge_data()
    expected_columns = {
        pG.edge_id_col_name,
        pG.src_col_name,
        pG.dst_col_name,
        pG.type_col_name,
        "vec1",
    }
    assert set(df.columns) == expected_columns
    expected = df1[["feat_0", "feat_1", "feat_2"]].values
    expected = expected[np.lexsort(expected.T)]  # may be jumbled, so sort

    pGalt = MGPropertyGraph()
    pGalt.add_edge_data(dd1, ("src", "dst"), vector_property="vec1")
    dfalt = pG.get_edge_data()

    for cur_pG, cur_df in [(pG, df), (pGalt, dfalt)]:
        vec1 = cur_pG.edge_vector_property_to_array(cur_df, "vec1").compute()
        vec1 = vec1[np.lexsort(vec1.T)]  # may be jumbled, so sort
        assert_array_equal(vec1, expected)
        vec1 = cur_pG.edge_vector_property_to_array(
            cur_df, "vec1", missing="error"
        ).compute()
        vec1 = vec1[np.lexsort(vec1.T)]  # may be jumbled, so sort
        assert_array_equal(vec1, expected)

    pG.add_edge_data(
        dd2, ("src", "dst"), vector_properties={"vec2": ["feat_0", "feat_1"]}
    )
    df = pG.get_edge_data()
    expected_columns.add("vec2")
    assert set(df.columns) == expected_columns
    expected = df2[["feat_0", "feat_1"]].values
    expected = expected[np.lexsort(expected.T)]  # may be jumbled, so sort
    vec2 = pG.edge_vector_property_to_array(df, "vec2").compute()
    vec2 = vec2[np.lexsort(vec2.T)]  # may be jumbled, so sort
    assert_array_equal(vec2, expected)
    with pytest.raises(RuntimeError):
        pG.edge_vector_property_to_array(df, "vec2", missing="error").compute()


def test_fillna_vertices():
    from cugraph.experimental import MGPropertyGraph

    df_edgelist = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "src": [0, 7, 2, 0, 1, 3, 1, 4, 5, 6],
                "dst": [1, 1, 1, 3, 2, 1, 6, 5, 6, 7],
                "val": [1, None, 2, None, 3, None, 4, None, 5, None],
            }
        ),
        npartitions=2,
    )

    df_props = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6, 7],
                "a": [0, 1, None, 2, None, 4, 1, 8],
                "b": [None, 1, None, 2, None, 3, 8, 9],
            }
        ),
        npartitions=2,
    )

    pG = MGPropertyGraph()
    pG.add_edge_data(df_edgelist, vertex_col_names=["src", "dst"])
    pG.add_vertex_data(df_props, vertex_col_name="id")

    pG.fillna_vertices({"a": 2, "b": 3})

    assert not pG.get_vertex_data(columns=["a", "b"]).compute().isna().any().any()
    assert pG.get_edge_data(columns=["val"]).compute().isna().any().any()

    expected_values_prop_a = [
        0,
        1,
        2,
        2,
        2,
        4,
        1,
        8,
    ]
    assert pG.get_vertex_data(columns=["a"])["a"].compute().values_host.tolist() == (
        expected_values_prop_a
    )

    expected_values_prop_b = [
        3,
        1,
        3,
        2,
        3,
        3,
        8,
        9,
    ]
    assert pG.get_vertex_data(columns=["b"])["b"].compute().values_host.tolist() == (
        expected_values_prop_b
    )


def test_fillna_edges():
    from cugraph.experimental import MGPropertyGraph

    df_edgelist = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "src": [0, 7, 2, 0, 1, 3, 1, 4, 5, 6],
                "dst": [1, 1, 1, 3, 2, 1, 6, 5, 6, 7],
                "val": [1, None, 2, None, 3, None, 4, None, 5, None],
            }
        ),
        npartitions=2,
    )

    df_props = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6, 7],
                "a": [0, 1, None, 2, None, 4, 1, 8],
                "b": [None, 1, None, 2, None, 3, 8, 9],
            }
        ),
        npartitions=2,
    )

    pG = MGPropertyGraph()
    pG.add_edge_data(df_edgelist, vertex_col_names=["src", "dst"])
    pG.add_vertex_data(df_props, vertex_col_name="id")

    pG.fillna_edges(2)

    assert not pG.get_edge_data(columns=["val"]).compute().isna().any().any()
    assert pG.get_vertex_data(columns=["a", "b"]).compute().isna().any().any()

    expected_values_prop_val = [
        1,
        2,
        2,
        2,
        3,
        2,
        4,
        2,
        5,
        2,
    ]
    assert pG.get_edge_data(columns=["val"])["val"].compute().values_host.tolist() == (
        expected_values_prop_val
    )


def test_types_from_numerals(dask_client):
    from cugraph.experimental import MGPropertyGraph

    df_edgelist_cow = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "src": [0, 7, 2, 0, 1],
                "dst": [1, 1, 1, 3, 2],
                "val": [1, 3, 2, 3, 3],
            }
        ),
        npartitions=2,
    )

    df_edgelist_pig = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "src": [3, 1, 4, 5, 6],
                "dst": [1, 6, 5, 6, 7],
                "val": [5, 4, 5, 5, 2],
            }
        ),
        npartitions=2,
    )

    df_props_duck = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "a": [0, 1, 6, 2],
                "b": [2, 1, 2, 2],
            }
        ),
        npartitions=2,
    )

    df_props_goose = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "id": [4, 5, 6, 7],
                "a": [5, 4, 1, 8],
                "b": [2, 3, 8, 9],
            }
        ),
        npartitions=2,
    )

    pG = MGPropertyGraph()

    pG.add_edge_data(df_edgelist_cow, vertex_col_names=["src", "dst"], type_name="cow")
    pG.add_edge_data(df_edgelist_pig, vertex_col_names=["src", "dst"], type_name="pig")

    pG.add_vertex_data(df_props_duck, vertex_col_name="id", type_name="duck")
    pG.add_vertex_data(df_props_goose, vertex_col_name="id", type_name="goose")

    assert pG.vertex_types_from_numerals(
        cudf.Series([0, 1, 0, 0, 1, 0, 1, 1])
    ).values_host.tolist() == [
        "duck",
        "goose",
        "duck",
        "duck",
        "goose",
        "duck",
        "goose",
        "goose",
    ]
    assert pG.edge_types_from_numerals(
        cudf.Series([1, 1, 0, 1, 1, 0, 0, 1, 1])
    ).values_host.tolist() == [
        "pig",
        "pig",
        "cow",
        "pig",
        "pig",
        "cow",
        "cow",
        "pig",
        "pig",
    ]


# =============================================================================
# Benchmarks
# =============================================================================


@pytest.mark.slow
@pytest.mark.parametrize("N", [1, 3, 10, 30])
def bench_add_edges_cyber(gpubenchmark, dask_client, N):
    from cugraph.experimental import MGPropertyGraph

    # Partition the dataframe to add in chunks
    cyber_df = cyber.get_edgelist()
    chunk = (len(cyber_df) + N - 1) // N
    dfs = [
        dask_cudf.from_cudf(cyber_df.iloc[i * chunk : (i + 1) * chunk], npartitions=2)
        for i in range(N)
    ]

    def func():
        mpG = MGPropertyGraph()
        for df in dfs:
            mpG.add_edge_data(df, ("srcip", "dstip"))
        df = mpG.get_edge_data().compute()
        assert len(df) == len(cyber_df)

    gpubenchmark(func)


@pytest.mark.slow
@pytest.mark.parametrize("n_rows", [1_000_000])
@pytest.mark.parametrize("n_feats", [128])
def bench_get_vector_features(gpubenchmark, dask_client, n_rows, n_feats):
    from cugraph.experimental import MGPropertyGraph

    df = cudf.DataFrame(
        {
            "src": cp.arange(0, n_rows, dtype=cp.int32),
            "dst": cp.arange(0, n_rows, dtype=cp.int32) + 1,
        }
    )
    for i in range(n_feats):
        df[f"feat_{i}"] = cp.ones(len(df), dtype=cp.int32)
    df = dask_cudf.from_cudf(df, npartitions=16)

    vector_properties = {"feat": [f"feat_{i}" for i in range(n_feats)]}
    pG = MGPropertyGraph()
    pG.add_edge_data(
        df, vertex_col_names=["src", "dst"], vector_properties=vector_properties
    )

    def func(pG):
        df = pG.get_edge_data(edge_ids=cp.arange(0, 100_000))
        df = df.compute()

    gpubenchmark(func, pG)
