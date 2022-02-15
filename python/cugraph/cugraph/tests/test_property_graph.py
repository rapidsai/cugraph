# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import time
import gc

import pytest
import pandas as pd
import numpy as np
import cudf
from cudf.testing import assert_frame_equal, assert_series_equal

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
from cugraph.generators import rmat
from cugraph.tests import utils

# =============================================================================
# Test data
# =============================================================================

dataset1 = {
    "merchants": [
        ["merchant_id", "merchant_location", "merchant_size", "merchant_sales",
         "merchant_num_employees", "merchant_name"],
        [(11, 78750, 44, 123.2, 12, "north"),
         (4, 78757, 112, 234.99, 18, "south"),
         (21, 44145, 83, 992.1, 27, "east"),
         (16, 47906, 92, 32.43, 5, "west"),
         (86, 47906, 192, 2.43, 51, "west"),
         ]
     ],
    "users": [
        ["user_id", "user_location", "vertical"],
        [(89021, 78757, 0),
         (32431, 78750, 1),
         (89216, 78757, 1),
         (78634, 47906, 0),
         ]
     ],
    "taxpayers": [
        ["payer_id", "amount"],
        [(11, 1123.98),
         (4, 3243.7),
         (21, 8932.3),
         (16, 3241.77),
         (86, 789.2),
         (89021, 23.98),
         (78634, 41.77),
         ]
    ],
    "transactions": [
        ["user_id", "merchant_id", "volume", "time", "card_num", "card_type"],
        [(89021, 11, 33.2, 1639084966.5513437, 123456, "MC"),
         (89216, 4, None, 1639085163.481217, 8832, "CASH"),
         (78634, 16, 72.0, 1639084912.567394, 4321, "DEBIT"),
         (32431, 4, 103.2, 1639084721.354346, 98124, "V"),
         ]
     ],
    "relationships": [
        ["user_id_1", "user_id_2", "relationship_type"],
        [(89216, 89021, 9),
         (89216, 32431, 9),
         (32431, 78634, 8),
         (78634, 89216, 8),
         ]
     ],
    "referrals": [
        ["user_id_1", "user_id_2", "merchant_id", "stars"],
        [(89216, 78634, 11, 5),
         (89021, 89216, 4, 4),
         (89021, 89216, 21, 3),
         (89021, 89216, 11, 3),
         (89021, 78634, 21, 4),
         (78634, 32431, 11, 4),
         ]
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
    DiGraph_inst = cugraph.Graph(directed=True)  # noqa: F841


# =============================================================================
# Pytest fixtures
# =============================================================================
df_types = [cudf.DataFrame, pd.DataFrame]


def df_type_id(dataframe_type):
    """
    Return a string that describes the dataframe_type, used for test output.
    """
    s = "df_type="
    if dataframe_type == cudf.DataFrame:
        return s+"cudf.DataFrame"
    if dataframe_type == pd.DataFrame:
        return s+"pandas.DataFrame"
    return s+"?"


df_types_fixture_params = utils.genFixtureParamsProduct((df_types, df_type_id))


@pytest.fixture(scope="module", params=df_types_fixture_params)
def dataset1_PropertyGraph(request):
    """
    Fixture which returns an instance of a PropertyGraph with vertex and edge
    data added from dataset1, parameterized for different DataFrame types.
    """
    dataframe_type = request.param[0]
    from cugraph.experimental import PropertyGraph

    (merchants, users, taxpayers,
     transactions, relationships, referrals) = dataset1.values()

    pG = PropertyGraph()

    # Vertex and edge data is added as one or more DataFrames; either a Pandas
    # DataFrame to keep data on the CPU, a cuDF DataFrame to keep data on GPU,
    # or a dask_cudf DataFrame to keep data on distributed GPUs.

    # For dataset1: vertices are merchants and users, edges are transactions,
    # relationships, and referrals.

    # property_columns=None (the default) means all columns except
    # vertex_col_name will be used as properties for the vertices/edges.

    pG.add_vertex_data(dataframe_type(columns=merchants[0],
                                      data=merchants[1]),
                       type_name="merchants",
                       vertex_col_name="merchant_id",
                       property_columns=None)
    pG.add_vertex_data(dataframe_type(columns=users[0],
                                      data=users[1]),
                       type_name="users",
                       vertex_col_name="user_id",
                       property_columns=None)
    pG.add_vertex_data(dataframe_type(columns=taxpayers[0],
                                      data=taxpayers[1]),
                       type_name="taxpayers",
                       vertex_col_name="payer_id",
                       property_columns=None)

    pG.add_edge_data(dataframe_type(columns=transactions[0],
                                    data=transactions[1]),
                     type_name="transactions",
                     vertex_col_names=("user_id", "merchant_id"),
                     property_columns=None)
    pG.add_edge_data(dataframe_type(columns=relationships[0],
                                    data=relationships[1]),
                     type_name="relationships",
                     vertex_col_names=("user_id_1", "user_id_2"),
                     property_columns=None)
    pG.add_edge_data(dataframe_type(columns=referrals[0],
                                    data=referrals[1]),
                     type_name="referrals",
                     vertex_col_names=("user_id_1",
                                       "user_id_2"),
                     property_columns=None)

    return pG


@pytest.fixture(scope="module", params=df_types_fixture_params)
def cyber_PropertyGraph(request):
    """
    Fixture which returns an instance of a PropertyGraph with vertex and edge
    data added from the cyber.csv dataset, parameterized for different
    DataFrame types.
    """
    from cugraph.experimental import PropertyGraph

    dataframe_type = request.param[0]
    cyber_csv = utils.RAPIDS_DATASET_ROOT_DIR_PATH/"cyber.csv"
    source_col_name = "srcip"
    dest_col_name = "dstip"

    if dataframe_type is pd.DataFrame:
        read_csv = pd.read_csv
    else:
        read_csv = cudf.read_csv
    df = read_csv(cyber_csv, delimiter=",",
                  dtype={"idx": "int32",
                         source_col_name: "str",
                         dest_col_name: "str"},
                  header=0)

    pG = PropertyGraph()
    pG.add_edge_data(df, (source_col_name, dest_col_name))

    return pG


@pytest.fixture(scope="module", params=df_types_fixture_params)
def rmat_PropertyGraph():
    """
    Fixture which uses the RMAT generator to generate a cuDF DataFrame
    edgelist, then uses it to add vertex and edge data to a PropertyGraph
    instance, then returns the (PropertyGraph, DataFrame) instances in a tuple.
    """
    from cugraph.experimental import PropertyGraph

    source_col_name = "src"
    dest_col_name = "dst"
    weight_col_name = "weight"
    scale = 20
    edgefactor = 16
    seed = 42
    df = rmat(scale,
              (2**scale)*edgefactor,
              0.57,  # from Graph500
              0.19,  # from Graph500
              0.19,  # from Graph500
              seed,
              clip_and_flip=False,
              scramble_vertex_ids=True,
              create_using=None,  # None == return edgelist
              mg=False
              )
    rng = np.random.default_rng(seed)
    df[weight_col_name] = rng.random(size=len(df))

    pG = PropertyGraph()
    pG.add_edge_data(df, (source_col_name, dest_col_name))

    return (pG, df)


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_add_vertex_data(df_type):
    """
    add_vertex_data() on "merchants" table, all properties.
    """
    from cugraph.experimental import PropertyGraph

    merchants = dataset1["merchants"]
    merchants_df = df_type(columns=merchants[0],
                           data=merchants[1])

    pG = PropertyGraph()
    pG.add_vertex_data(merchants_df,
                       type_name="merchants",
                       vertex_col_name="merchant_id",
                       property_columns=None)

    assert pG.num_vertices == 5
    assert pG.num_edges == 0
    expected_props = merchants[0].copy()
    assert sorted(pG.vertex_property_names) == sorted(expected_props)


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_num_vertices(df_type):
    """
    Ensures num_vertices is correct after various additions of specific data.
    """
    from cugraph.experimental import PropertyGraph

    merchants = dataset1["merchants"]
    merchants_df = df_type(columns=merchants[0],
                           data=merchants[1])

    pG = PropertyGraph()
    pG.add_vertex_data(merchants_df,
                       type_name="merchants",
                       vertex_col_name="merchant_id",
                       property_columns=None)

    # Test caching - the second retrieval should always be faster
    st = time.time()
    assert pG.num_vertices == 5
    compute_time = time.time() - st
    assert pG.num_edges == 0

    st = time.time()
    assert pG.num_vertices == 5
    cache_retrieval_time = time.time() - st
    assert cache_retrieval_time < compute_time

    users = dataset1["users"]
    users_df = df_type(columns=users[0], data=users[1])

    pG.add_vertex_data(users_df,
                       type_name="users",
                       vertex_col_name="user_id",
                       property_columns=None)

    assert pG.num_vertices == 9
    assert pG.num_edges == 0

    # The taxpayers table does not add new vertices, it only adds properties to
    # vertices already present in the merchants and users tables.
    taxpayers = dataset1["taxpayers"]
    taxpayers_df = df_type(columns=taxpayers[0],
                           data=taxpayers[1])

    pG.add_vertex_data(taxpayers_df,
                       type_name="taxpayers",
                       vertex_col_name="payer_id",
                       property_columns=None)

    assert pG.num_vertices == 9
    assert pG.num_edges == 0


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_null_data(df_type):
    """
    test for null data
    """
    from cugraph.experimental import PropertyGraph

    pG = PropertyGraph()

    assert pG.num_vertices == 0
    assert pG.num_edges == 0
    assert sorted(pG.vertex_property_names) == sorted([])


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_add_vertex_data_prop_columns(df_type):
    """
    add_vertex_data() on "merchants" table, subset of properties.
    """
    from cugraph.experimental import PropertyGraph

    merchants = dataset1["merchants"]
    merchants_df = df_type(columns=merchants[0],
                           data=merchants[1])
    expected_props = ["merchant_name", "merchant_sales", "merchant_location"]

    pG = PropertyGraph()
    pG.add_vertex_data(merchants_df,
                       type_name="merchants",
                       vertex_col_name="merchant_id",
                       property_columns=expected_props)

    assert pG.num_vertices == 5
    assert pG.num_edges == 0
    assert sorted(pG.vertex_property_names) == sorted(expected_props)


def test_add_vertex_data_bad_args():
    """
    add_vertex_data() with various bad args, checks that proper exceptions are
    raised.
    """
    from cugraph.experimental import PropertyGraph

    merchants = dataset1["merchants"]
    merchants_df = cudf.DataFrame(columns=merchants[0],
                                  data=merchants[1])

    pG = PropertyGraph()
    with pytest.raises(TypeError):
        pG.add_vertex_data(42,
                           type_name="merchants",
                           vertex_col_name="merchant_id",
                           property_columns=None)
    with pytest.raises(TypeError):
        pG.add_vertex_data(merchants_df,
                           type_name=42,
                           vertex_col_name="merchant_id",
                           property_columns=None)
    with pytest.raises(ValueError):
        pG.add_vertex_data(merchants_df,
                           type_name="merchants",
                           vertex_col_name="bad_column_name",
                           property_columns=None)
    with pytest.raises(ValueError):
        pG.add_vertex_data(merchants_df,
                           type_name="merchants",
                           vertex_col_name="merchant_id",
                           property_columns=["bad_column_name",
                                             "merchant_name"])
    with pytest.raises(TypeError):
        pG.add_vertex_data(merchants_df,
                           type_name="merchants",
                           vertex_col_name="merchant_id",
                           property_columns="merchant_name")


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_add_edge_data(df_type):
    """
    add_edge_data() on "transactions" table, all properties.
    """
    from cugraph.experimental import PropertyGraph

    transactions = dataset1["transactions"]
    transactions_df = df_type(columns=transactions[0],
                              data=transactions[1])

    pG = PropertyGraph()
    pG.add_edge_data(transactions_df,
                     type_name="transactions",
                     vertex_col_names=("user_id", "merchant_id"),
                     property_columns=None)

    assert pG.num_vertices == 7
    assert pG.num_edges == 4
    expected_props = ["merchant_id", "user_id",
                      "volume", "time", "card_num", "card_type"]
    assert sorted(pG.edge_property_names) == sorted(expected_props)


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_add_edge_data_prop_columns(df_type):
    """
    add_edge_data() on "transactions" table, subset of properties.
    """
    from cugraph.experimental import PropertyGraph

    transactions = dataset1["transactions"]
    transactions_df = df_type(columns=transactions[0],
                              data=transactions[1])
    expected_props = ["card_num", "card_type"]

    pG = PropertyGraph()
    pG.add_edge_data(transactions_df,
                     type_name="transactions",
                     vertex_col_names=("user_id", "merchant_id"),
                     property_columns=expected_props)

    assert pG.num_vertices == 7
    assert pG.num_edges == 4
    assert sorted(pG.edge_property_names) == sorted(expected_props)


def test_add_edge_data_bad_args():
    """
    add_edge_data() with various bad args, checks that proper exceptions are
    raised.
    """
    from cugraph.experimental import PropertyGraph

    transactions = dataset1["transactions"]
    transactions_df = cudf.DataFrame(columns=transactions[0],
                                     data=transactions[1])

    pG = PropertyGraph()
    with pytest.raises(TypeError):
        pG.add_edge_data(42,
                         type_name="transactions",
                         vertex_col_names=("user_id", "merchant_id"),
                         property_columns=None)
    with pytest.raises(TypeError):
        pG.add_edge_data(transactions_df,
                         type_name=42,
                         vertex_col_names=("user_id", "merchant_id"),
                         property_columns=None)
    with pytest.raises(ValueError):
        pG.add_edge_data(transactions_df,
                         type_name="transactions",
                         vertex_col_names=("user_id", "bad_column"),
                         property_columns=None)
    with pytest.raises(ValueError):
        pG.add_edge_data(transactions_df,
                         type_name="transactions",
                         vertex_col_names=("user_id", "merchant_id"),
                         property_columns=["bad_column_name", "time"])
    with pytest.raises(TypeError):
        pG.add_edge_data(transactions_df,
                         type_name="transactions",
                         vertex_col_names=("user_id", "merchant_id"),
                         property_columns="time")


def test_extract_subgraph_vertex_prop_condition_only(dataset1_PropertyGraph):

    pG = dataset1_PropertyGraph

    selection = pG.select_vertices("(_TYPE_=='taxpayers') & (amount<100)")
    G = pG.extract_subgraph(selection=selection,
                            create_using=DiGraph_inst,
                            edge_weight_property="stars")

    expected_edgelist = cudf.DataFrame({"src": [89021], "dst": [78634],
                                        "weights": [4]})
    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src",
                                   preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst",
                                   preserve_order=True)

    assert G.is_directed()
    # check_like=True ignores differences in column/index ordering
    assert_frame_equal(expected_edgelist, actual_edgelist, check_like=True)


def test_extract_subgraph_vertex_edge_prop_condition(dataset1_PropertyGraph):
    from cugraph.experimental import PropertyGraph

    pG = dataset1_PropertyGraph
    tcn = PropertyGraph.type_col_name

    selection = pG.select_vertices("(user_location==47906) | "
                                   "(user_location==78750)")
    selection += pG.select_edges(f"{tcn}=='referrals'")
    G = pG.extract_subgraph(selection=selection,
                            create_using=DiGraph_inst,
                            edge_weight_property="stars")

    expected_edgelist = cudf.DataFrame({"src": [78634], "dst": [32431],
                                        "weights": [4]})
    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src",
                                   preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst",
                                   preserve_order=True)

    assert G.is_directed()
    assert_frame_equal(expected_edgelist, actual_edgelist, check_like=True)


def test_extract_subgraph_edge_prop_condition_only(dataset1_PropertyGraph):
    from cugraph.experimental import PropertyGraph

    pG = dataset1_PropertyGraph
    tcn = PropertyGraph.type_col_name

    selection = pG.select_edges(f"{tcn} =='transactions'")
    G = pG.extract_subgraph(selection=selection,
                            create_using=DiGraph_inst)

    # last item is the DataFrame rows
    transactions = dataset1["transactions"][-1]
    (srcs, dsts) = zip(*[(t[0], t[1]) for t in transactions])
    expected_edgelist = cudf.DataFrame({"src": srcs, "dst": dsts})
    expected_edgelist = expected_edgelist.sort_values(by="src",
                                                      ignore_index=True)

    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src",
                                   preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst",
                                   preserve_order=True)
    actual_edgelist = actual_edgelist.sort_values(by="src", ignore_index=True)

    assert G.is_directed()
    assert_frame_equal(expected_edgelist, actual_edgelist, check_like=True)


def test_extract_subgraph_unweighted(dataset1_PropertyGraph):
    """
    Ensure a subgraph is unweighted if the edge_weight_property is None.
    """
    from cugraph.experimental import PropertyGraph

    pG = dataset1_PropertyGraph
    tcn = PropertyGraph.type_col_name

    selection = pG.select_edges(f"{tcn} == 'transactions'")
    G = pG.extract_subgraph(selection=selection,
                            create_using=DiGraph_inst)

    assert G.is_weighted() is False


def test_extract_subgraph_specific_query(dataset1_PropertyGraph):
    """
    Graph of only transactions after time 1639085000 for merchant_id 4 (should
    be a graph of 2 vertices, 1 edge)
    """
    from cugraph.experimental import PropertyGraph

    pG = dataset1_PropertyGraph
    tcn = PropertyGraph.type_col_name

    selection = pG.select_edges(f"({tcn}=='transactions') & "
                                "(merchant_id==4) & "
                                "(time>1639085000)")
    G = pG.extract_subgraph(selection=selection,
                            create_using=DiGraph_inst,
                            edge_weight_property="card_num")

    expected_edgelist = cudf.DataFrame({"src": [89216], "dst": [4],
                                        "weights": [8832]})
    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src",
                                   preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst",
                                   preserve_order=True)

    assert G.is_directed()
    assert_frame_equal(expected_edgelist, actual_edgelist, check_like=True)


def test_edge_props_to_graph(dataset1_PropertyGraph):
    """
    Access the property DataFrames directly and use them to perform a more
    complex query, then call edge_props_to_graph() to create the corresponding
    graph.
    """
    from cugraph.experimental import PropertyGraph

    pG = dataset1_PropertyGraph
    vcn = PropertyGraph.vertex_col_name
    tcn = PropertyGraph.type_col_name
    scn = PropertyGraph.src_col_name
    dcn = PropertyGraph.dst_col_name

    # Select referrals from only taxpayers who are users (should be 1)

    # Find the list of vertices that are both users and taxpayers
    def contains_both(df):
        return (df[tcn] == "taxpayers").any() and \
            (df[tcn] == "users").any()
    verts = pG._vertex_prop_dataframe.groupby(vcn)\
                                     .apply(contains_both)
    verts = verts[verts].keys()  # get an array of only verts that have both

    # Find the "referral" edge_props containing only those verts
    referrals = pG._edge_prop_dataframe[tcn] == "referrals"
    srcs = pG._edge_prop_dataframe[referrals][scn].isin(verts)
    dsts = pG._edge_prop_dataframe[referrals][dcn].isin(verts)
    matching_edges = (srcs & dsts)
    indices = matching_edges.index[matching_edges]
    edge_props = pG._edge_prop_dataframe.loc[indices]

    G = pG.edge_props_to_graph(edge_props,
                               create_using=DiGraph_inst)

    expected_edgelist = cudf.DataFrame({"src": [89021], "dst": [78634]})
    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src",
                                   preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst",
                                   preserve_order=True)

    assert G.is_directed()
    assert_frame_equal(expected_edgelist, actual_edgelist, check_like=True)


def test_select_vertices_from_previous_selection(dataset1_PropertyGraph):
    """
    Ensures that the intersection of vertices of multiple types (only vertices
    that are both type A and type B) can be selected.
    """
    from cugraph.experimental import PropertyGraph

    pG = dataset1_PropertyGraph
    tcn = PropertyGraph.type_col_name

    # Select referrals from only taxpayers who are users (should be 1)
    selection = pG.select_vertices(f"{tcn} == 'taxpayers'")
    selection = pG.select_vertices(f"{tcn} == 'users'",
                                   from_previous_selection=selection)
    selection += pG.select_edges(f"{tcn} == 'referrals'")
    G = pG.extract_subgraph(create_using=DiGraph_inst, selection=selection)

    expected_edgelist = cudf.DataFrame({"src": [89021], "dst": [78634]})
    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src",
                                   preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst",
                                   preserve_order=True)

    assert G.is_directed()
    assert_frame_equal(expected_edgelist, actual_edgelist, check_like=True)


def test_extract_subgraph_graph_without_vert_props():
    """
    Ensure a subgraph can be extracted from a PropertyGraph that does not have
    vertex properties.
    """
    from cugraph.experimental import PropertyGraph

    transactions = dataset1["transactions"]
    relationships = dataset1["relationships"]

    pG = PropertyGraph()

    pG.add_edge_data(cudf.DataFrame(columns=transactions[0],
                                    data=transactions[1]),
                     type_name="transactions",
                     vertex_col_names=("user_id", "merchant_id"),
                     property_columns=None)
    pG.add_edge_data(cudf.DataFrame(columns=relationships[0],
                                    data=relationships[1]),
                     type_name="relationships",
                     vertex_col_names=("user_id_1", "user_id_2"),
                     property_columns=None)

    scn = PropertyGraph.src_col_name
    G = pG.extract_subgraph(selection=pG.select_edges(f"{scn} == 89216"),
                            create_using=DiGraph_inst,
                            edge_weight_property="relationship_type",
                            default_edge_weight=0)

    expected_edgelist = cudf.DataFrame({"src": [89216, 89216, 89216],
                                        "dst": [4, 89021, 32431],
                                        "weights": [0, 9, 9]})
    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src",
                                   preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst",
                                   preserve_order=True)

    assert G.is_directed()
    assert_frame_equal(expected_edgelist, actual_edgelist, check_like=True)


def test_extract_subgraph_no_edges(dataset1_PropertyGraph):
    """
    Valid query that only matches a single vertex.
    """
    pG = dataset1_PropertyGraph

    selection = pG.select_vertices("(_TYPE_=='merchants') & (merchant_id==86)")
    G = pG.extract_subgraph(selection=selection)

    assert len(G.edgelist.edgelist_df) == 0


def test_extract_subgraph_no_query(dataset1_PropertyGraph):
    """
    Call extract with no args, should result in the entire property graph.
    """
    pG = dataset1_PropertyGraph

    G = pG.extract_subgraph(create_using=DiGraph_inst, allow_multi_edges=True)

    num_edges = \
        len(dataset1["transactions"][-1]) + \
        len(dataset1["relationships"][-1]) + \
        len(dataset1["referrals"][-1])
    # referrals has 3 edges with the same src/dst, so subtract 2 from
    # the total count since this is not creating a multigraph..
    num_edges -= 2
    assert len(G.edgelist.edgelist_df) == num_edges


def test_extract_subgraph_multi_edges(dataset1_PropertyGraph):
    """
    Ensure an exception is thrown if a graph is attempted to be extracted with
    multi edges.
    NOTE: an option to allow multi edges when create_using is
    MultiGraph will be provided in the future.
    """
    from cugraph.experimental import PropertyGraph

    pG = dataset1_PropertyGraph
    tcn = PropertyGraph.type_col_name

    # referrals has multiple edges
    selection = pG.select_edges(f"{tcn} == 'referrals'")

    # FIXME: use a better exception
    with pytest.raises(RuntimeError):
        pG.extract_subgraph(selection=selection,
                            create_using=DiGraph_inst)


def test_extract_subgraph_bad_args(dataset1_PropertyGraph):
    from cugraph.experimental import PropertyGraph

    pG = dataset1_PropertyGraph
    tcn = PropertyGraph.type_col_name

    # non-PropertySelection selection
    with pytest.raises(TypeError):
        pG.extract_subgraph(selection=78750,
                            create_using=DiGraph_inst,
                            edge_weight_property="stars",
                            default_edge_weight=1.0)

    selection = pG.select_edges(f"{tcn}=='referrals'")
    # bad create_using type
    with pytest.raises(TypeError):
        pG.extract_subgraph(selection=selection,
                            create_using=pytest,
                            edge_weight_property="stars",
                            default_edge_weight=1.0)
    # invalid column name
    with pytest.raises(ValueError):
        pG.extract_subgraph(selection=selection,
                            edge_weight_property="bad_column",
                            default_edge_weight=1.0)
    # column name has None value for all results in subgraph and
    # default_edge_weight is not set.
    with pytest.raises(ValueError):
        pG.extract_subgraph(selection=selection,
                            edge_weight_property="card_type")


def test_extract_subgraph_default_edge_weight(dataset1_PropertyGraph):
    """
    Ensure the default_edge_weight value is added to edges with missing
    properties used for weights.
    """
    from cugraph.experimental import PropertyGraph

    pG = dataset1_PropertyGraph
    tcn = PropertyGraph.type_col_name

    selection = pG.select_edges(f"{tcn}=='transactions'")
    G = pG.extract_subgraph(create_using=DiGraph_inst,
                            selection=selection,
                            edge_weight_property="volume",
                            default_edge_weight=99)

    # last item is the DataFrame rows
    transactions = dataset1["transactions"][-1]
    (srcs, dsts, weights) = zip(*[(t[0], t[1], t[2])
                                  for t in transactions])
    # replace None with the expected value (convert to a list to replace)
    weights_list = list(weights)
    weights_list[weights.index(None)] = 99.
    weights = tuple(weights_list)
    expected_edgelist = cudf.DataFrame({"src": srcs, "dst": dsts,
                                        "weights": weights})
    expected_edgelist = expected_edgelist.sort_values(by="src",
                                                      ignore_index=True)

    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src",
                                   preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst",
                                   preserve_order=True)
    actual_edgelist = actual_edgelist.sort_values(by="src",
                                                  ignore_index=True)

    assert G.is_directed()
    assert_frame_equal(expected_edgelist, actual_edgelist, check_like=True)


def test_extract_subgraph_default_edge_weight_no_property(
        dataset1_PropertyGraph):
    """
    Ensure default_edge_weight can be used to provide an edge value when a
    property for the edge weight is not specified.
    """
    pG = dataset1_PropertyGraph
    edge_weight = 99.2
    G = pG.extract_subgraph(allow_multi_edges=True,
                            default_edge_weight=edge_weight)
    assert (G.edgelist.edgelist_df["weights"] == edge_weight).all()


def test_graph_edge_data_added(dataset1_PropertyGraph):
    """
    Ensures the subgraph returned from extract_subgraph() has the edge_data
    attribute added which contains the proper edge IDs.
    """
    from cugraph.experimental import PropertyGraph

    pG = dataset1_PropertyGraph
    eicn = PropertyGraph.edge_id_col_name

    expected_num_edges = \
        len(dataset1["transactions"][-1]) + \
        len(dataset1["relationships"][-1]) + \
        len(dataset1["referrals"][-1])

    assert pG.num_edges == expected_num_edges

    # extract_subgraph() should return a directed Graph object with additional
    # meta-data, which includes edge IDs.
    G = pG.extract_subgraph(create_using=DiGraph_inst, allow_multi_edges=True)

    # G.edge_data should be set to a DataFrame with rows for each graph edge.
    assert len(G.edge_data) == expected_num_edges
    edge_ids = sorted(G.edge_data[eicn].values)

    assert edge_ids[0] == 0
    assert edge_ids[-1] == (expected_num_edges - 1)


def test_annotate_dataframe(dataset1_PropertyGraph):
    """
    FIXME: Add tests for:
    properties list
    properties list with 1 or more bad props
    copy=False
    invalid args raise correct exceptions
    """
    pG = dataset1_PropertyGraph

    selection = pG.select_edges("(_TYPE_ == 'referrals') & (stars > 3)")
    G = pG.extract_subgraph(selection=selection,
                            create_using=DiGraph_inst)

    df_type = type(pG._edge_prop_dataframe)
    # Create an arbitrary DataFrame meant to represent an algo result,
    # containing vertex IDs present in pG.
    #
    # Drop duplicate edges since actual results from a Graph object would not
    # have them.
    (srcs, dsts, mids, stars) = zip(*(dataset1["referrals"][1]))
    algo_result = df_type({"from": srcs, "to": dsts,
                           "result": range(len(srcs))})
    algo_result.drop_duplicates(subset=["from", "to"],
                                inplace=True, ignore_index=True)

    new_algo_result = pG.annotate_dataframe(
        algo_result, G, edge_vertex_col_names=("from", "to"))
    expected_algo_result = df_type({"from": srcs, "to": dsts,
                                    "result": range(len(srcs)),
                                    "merchant_id": mids,
                                    "stars": stars})
    # The integer dtypes of annotated properties are nullable integer dtypes,
    # so convert for proper comparison.
    expected_algo_result["merchant_id"] = \
        expected_algo_result["merchant_id"].astype("Int64")
    expected_algo_result["stars"] = \
        expected_algo_result["stars"].astype("Int64")

    expected_algo_result.drop_duplicates(subset=["from", "to"],
                                         inplace=True, ignore_index=True)

    if df_type is cudf.DataFrame:
        ase = assert_series_equal
    else:
        ase = pd.testing.assert_series_equal
    # For now, the result will include extra columns from edge types not
    # included in the df being annotated, so just check for known columns.
    for col in ["from", "to", "result", "merchant_id", "stars"]:
        ase(new_algo_result[col], expected_algo_result[col])


def test_different_vertex_edge_input_dataframe_types():
    """
    Ensures that a PropertyGraph initialized with one DataFrame type cannot be
    extended with another.
    """
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    from cugraph.experimental import PropertyGraph

    pG = PropertyGraph()
    pG.add_vertex_data(df, type_name="foo", vertex_col_name="a")
    with pytest.raises(TypeError):
        pG.add_edge_data(pdf, type_name="bar", vertex_col_names=("a", "b"))

    pG = PropertyGraph()
    pG.add_vertex_data(pdf, type_name="foo", vertex_col_name="a")
    with pytest.raises(TypeError):
        pG.add_edge_data(df, type_name="bar", vertex_col_names=("a", "b"))

    # Different order
    pG = PropertyGraph()
    pG.add_edge_data(df, type_name="bar", vertex_col_names=("a", "b"))
    with pytest.raises(TypeError):
        pG.add_vertex_data(pdf, type_name="foo", vertex_col_name="a")

    # Same API call, different types
    pG = PropertyGraph()
    pG.add_vertex_data(df, type_name="foo", vertex_col_name="a")
    with pytest.raises(TypeError):
        pG.add_vertex_data(pdf, type_name="foo", vertex_col_name="a")

    pG = PropertyGraph()
    pG.add_edge_data(df, type_name="bar", vertex_col_names=("a", "b"))
    with pytest.raises(TypeError):
        pG.add_edge_data(pdf, type_name="bar", vertex_col_names=("a", "b"))


def test_get_vertices(dataset1_PropertyGraph):
    """
    Test that get_vertices() returns the correct set of vertices without
    duplicates.
    """
    pG = dataset1_PropertyGraph

    (merchants, users, taxpayers,
     transactions, relationships, referrals) = dataset1.values()

    expected_vertices = set([t[0] for t in merchants[1]] +
                            [t[0] for t in users[1]] +
                            [t[0] for t in taxpayers[1]])

    assert sorted(pG.get_vertices().values) == sorted(expected_vertices)


def test_get_edges(dataset1_PropertyGraph):
    """
    Test that get_edges() returns the correct set of edges (as src/dst
    columns).
    """
    from cugraph.experimental import PropertyGraph

    pG = dataset1_PropertyGraph

    (merchants, users, taxpayers,
     transactions, relationships, referrals) = dataset1.values()

    expected_edges = \
        [(src, dst) for (src, dst, _, _, _, _) in transactions[1]] + \
        [(src, dst) for (src, dst, _) in relationships[1]] + \
        [(src, dst) for (src, dst, _, _) in referrals[1]]

    actual_edges = pG.edges

    assert len(expected_edges) == len(actual_edges)
    for i in range(len(expected_edges)):
        src = actual_edges[PropertyGraph.src_col_name].iloc[i]
        dst = actual_edges[PropertyGraph.dst_col_name].iloc[i]
        assert (src, dst) in expected_edges


@pytest.mark.skip(reason="unfinished")
def test_extract_subgraph_with_vertex_ids():
    """
    FIXME: add a PropertyGraph API that makes it easy to support the common use
    case of extracting a subgraph containing only specific vertex IDs. This is
    currently done in the bench_extract_subgraph_for_* tests below, but could
    be made easier for users to do.
    """
    raise NotImplementedError


@pytest.mark.skip(reason="unfinished")
def test_dgl_use_case():
    """
    FIXME: add a test demonstrating typical DGL use cases
    """
    raise NotImplementedError


# =============================================================================
# Benchmarks
# =============================================================================
def bench_num_vertices(gpubenchmark, dataset1_PropertyGraph):
    pG = dataset1_PropertyGraph

    def get_num_vertices():
        return pG.num_vertices

    assert gpubenchmark(get_num_vertices) == 9


def bench_get_vertices(gpubenchmark, dataset1_PropertyGraph):
    pG = dataset1_PropertyGraph

    gpubenchmark(pG.get_vertices)


def bench_extract_subgraph_for_cyber(gpubenchmark, cyber_PropertyGraph):
    from cugraph.experimental import PropertyGraph

    pG = cyber_PropertyGraph
    scn = PropertyGraph.src_col_name
    dcn = PropertyGraph.dst_col_name

    # Create a Graph containing only specific src or dst vertices
    verts = ["10.40.182.3", "10.40.182.255", "59.166.0.9", "59.166.0.8"]
    selected_edges = \
        pG.select_edges(f"{scn}.isin({verts}) | {dcn}.isin({verts})")
    gpubenchmark(pG.extract_subgraph,
                 create_using=cugraph.Graph(directed=True),
                 selection=selected_edges,
                 default_edge_weight=1.0,
                 allow_multi_edges=True)


def bench_extract_subgraph_for_cyber_detect_duplicate_edges(
        gpubenchmark, cyber_PropertyGraph):
    from cugraph.experimental import PropertyGraph

    pG = cyber_PropertyGraph
    scn = PropertyGraph.src_col_name
    dcn = PropertyGraph.dst_col_name

    # Create a Graph containing only specific src or dst vertices
    verts = ["10.40.182.3", "10.40.182.255", "59.166.0.9", "59.166.0.8"]
    selected_edges = \
        pG.select_edges(f"{scn}.isin({verts}) | {dcn}.isin({verts})")

    def func():
        with pytest.raises(RuntimeError):
            pG.extract_subgraph(create_using=cugraph.Graph(directed=True),
                                selection=selected_edges,
                                default_edge_weight=1.0,
                                allow_multi_edges=False)

    gpubenchmark(func)


def bench_extract_subgraph_for_rmat(gpubenchmark, rmat_PropertyGraph):
    from cugraph.experimental import PropertyGraph

    (pG, generated_df) = rmat_PropertyGraph
    scn = PropertyGraph.src_col_name
    dcn = PropertyGraph.dst_col_name

    verts = []
    for i in range(0, 10000, 10):
        verts.append(generated_df["src"].iloc[i])

    selected_edges = \
        pG.select_edges(f"{scn}.isin({verts}) | {dcn}.isin({verts})")
    gpubenchmark(pG.extract_subgraph,
                 create_using=cugraph.Graph(directed=True),
                 selection=selected_edges,
                 default_edge_weight=1.0,
                 allow_multi_edges=True)


# This test runs for *minutes* with the current implementation, and since
# benchmarking can call it multiple times per run, the overall time for this
# test can be ~20 minutes.
@pytest.mark.slow
def bench_extract_subgraph_for_rmat_detect_duplicate_edges(
        gpubenchmark, rmat_PropertyGraph):
    from cugraph.experimental import PropertyGraph

    (pG, generated_df) = rmat_PropertyGraph
    scn = PropertyGraph.src_col_name
    dcn = PropertyGraph.dst_col_name

    verts = []
    for i in range(0, 10000, 10):
        verts.append(generated_df["src"].iloc[i])

    selected_edges = \
        pG.select_edges(f"{scn}.isin({verts}) | {dcn}.isin({verts})")

    def func():
        with pytest.raises(RuntimeError):
            pG.extract_subgraph(create_using=cugraph.Graph(directed=True),
                                selection=selected_edges,
                                default_edge_weight=1.0,
                                allow_multi_edges=False)

    gpubenchmark(func)
