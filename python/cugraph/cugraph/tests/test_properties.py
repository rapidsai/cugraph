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

import gc

import pytest
import pandas as pd
import cudf
from cudf.testing import assert_frame_equal

import cugraph
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
    "transactions": [
        ["user_id", "merchant_id", "volume", "time", "card_num", "card_type"],
        [(89021, 11, 33.2, 1639084966.5513437, 123456, "MC"),
         (89216, 4, 12.8, 1639085163.481217, None, "CASH"),
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
         (32431, 89216, 4, 4),
         (89021, 78634, 21, 4),
         (78634, 89216, 11, 4),
         ]
     ],
}


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Pytest fixtures
# =============================================================================
df_types = [cudf.DataFrame, pd.DataFrame]
df_types = [cudf.DataFrame]


def df_type_id(dft):
    s = "df_type="
    if dft == cudf.DataFrame:
        return s+"cudf.DataFrame"
    if dft == pd.DataFrame:
        return s+"pandas.DataFrame"
    return s+"?"


@pytest.fixture(scope="module",
                params=utils.genFixtureParamsProduct((df_types, df_type_id))
                )
def property_graph_instance(request):
    """
    FIXME: fill this in
    """
    dataframe_type = request.param[0]
    from cugraph import PropertyGraph

    (merchants, users,
     transactions, relationships, referrals) = dataset1.values()

    pG = PropertyGraph()

    # Vertex and edge data is added as one or more DataFrames; either a Pandas
    # DataFrame to keep data on the CPU, a cuDF DataFrame to keep data on GPU,
    # or a dask_cudf DataFrame to keep data on distributed GPUs.

    # For dataset1: vertices are merchants and users, edges are transactions,
    # relationships, and referrals.

    # property_columns=None (the default) means all columns except
    # vertex_id_column will be used as properties for the vertices/edges.

    # FIXME: add a test for pandas DataFrames
    pG.add_vertex_data(dataframe_type(columns=merchants[0],
                                      data=merchants[1]),
                       type_name="merchants",
                       vertex_id_column="merchant_id",
                       property_columns=None)
    pG.add_vertex_data(dataframe_type(columns=users[0],
                                      data=users[1]),
                       type_name="users",
                       vertex_id_column="user_id",
                       property_columns=None)

    pG.add_edge_data(dataframe_type(columns=transactions[0],
                                    data=transactions[1]),
                     type_name="transactions",
                     vertex_id_columns=("user_id", "merchant_id"),
                     property_columns=None)
    pG.add_edge_data(dataframe_type(columns=relationships[0],
                                    data=relationships[1]),
                     type_name="relationships",
                     vertex_id_columns=("user_id_1", "user_id_2"),
                     property_columns=None)
    pG.add_edge_data(dataframe_type(columns=referrals[0],
                                    data=referrals[1]),
                     type_name="referrals",
                     vertex_id_columns=("user_id_1",
                                        "user_id_2"),
                     property_columns=None)

    return pG


###############################################################################
# Tests
@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_add_vertex_data(df_type):
    """
    add_vertex_data() on "merchants" table, all properties.
    """
    from cugraph import PropertyGraph

    merchants = dataset1["merchants"]
    merchants_df = df_type(columns=merchants[0],
                           data=merchants[1])

    pG = PropertyGraph()
    pG.add_vertex_data(merchants_df,
                       type_name="merchants",
                       vertex_id_column="merchant_id",
                       property_columns=None)

    assert pG.num_vertices == 5
    assert pG.num_edges == 0
    expected_props = merchants[0].copy()
    assert sorted(pG.vertex_property_names) == sorted(expected_props)


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_add_vertex_data_prop_columns(df_type):
    """
    add_vertex_data() on "merchants" table, subset of properties.
    """
    from cugraph import PropertyGraph

    merchants = dataset1["merchants"]
    merchants_df = df_type(columns=merchants[0],
                           data=merchants[1])
    expected_props = ["merchant_name", "merchant_sales", "merchant_location"]

    pG = PropertyGraph()
    pG.add_vertex_data(merchants_df,
                       type_name="merchants",
                       vertex_id_column="merchant_id",
                       property_columns=expected_props)

    assert pG.num_vertices == 5
    assert pG.num_edges == 0
    assert sorted(pG.vertex_property_names) == sorted(expected_props)


def test_add_vertex_data_bad_args():
    """
    add_vertex_data() with various bad args, checks that proper exceptions are
    raised.
    """
    from cugraph import PropertyGraph

    merchants = dataset1["merchants"]
    merchants_df = cudf.DataFrame(columns=merchants[0],
                                  data=merchants[1])

    pG = PropertyGraph()
    with pytest.raises(TypeError):
        pG.add_vertex_data(42,
                           type_name="merchants",
                           vertex_id_column="merchant_id",
                           property_columns=None)
    with pytest.raises(TypeError):
        pG.add_vertex_data(merchants_df,
                           type_name=42,
                           vertex_id_column="merchant_id",
                           property_columns=None)
    with pytest.raises(ValueError):
        pG.add_vertex_data(merchants_df,
                           type_name="merchants",
                           vertex_id_column="bad_column_name",
                           property_columns=None)
    with pytest.raises(ValueError):
        pG.add_vertex_data(merchants_df,
                           type_name="merchants",
                           vertex_id_column="merchant_id",
                           property_columns=["bad_column_name",
                                             "merchant_name"])
    with pytest.raises(TypeError):
        pG.add_vertex_data(merchants_df,
                           type_name="merchants",
                           vertex_id_column="merchant_id",
                           property_columns="merchant_name")


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_add_edge_data(df_type):
    """
    add_edge_data() on "transactions" table, all properties.
    """
    from cugraph import PropertyGraph

    transactions = dataset1["transactions"]
    transactions_df = df_type(columns=transactions[0],
                              data=transactions[1])

    pG = PropertyGraph()
    pG.add_edge_data(transactions_df,
                     type_name="transactions",
                     vertex_id_columns=("user_id", "merchant_id"),
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
    from cugraph import PropertyGraph

    transactions = dataset1["transactions"]
    transactions_df = df_type(columns=transactions[0],
                              data=transactions[1])
    expected_props = ["card_num", "card_type"]

    pG = PropertyGraph()
    pG.add_edge_data(transactions_df,
                     type_name="transactions",
                     vertex_id_columns=("user_id", "merchant_id"),
                     property_columns=expected_props)

    assert pG.num_vertices == 7
    assert pG.num_edges == 4
    assert sorted(pG.edge_property_names) == sorted(expected_props)


def test_add_edge_data_bad_args():
    """
    add_edge_data() with various bad args, checks that proper exceptions are
    raised.
    """
    from cugraph import PropertyGraph

    transactions = dataset1["transactions"]
    transactions_df = cudf.DataFrame(columns=transactions[0],
                                     data=transactions[1])

    pG = PropertyGraph()
    with pytest.raises(TypeError):
        pG.add_edge_data(42,
                         type_name="transactions",
                         vertex_id_columns=("user_id", "merchant_id"),
                         property_columns=None)
    with pytest.raises(TypeError):
        pG.add_edge_data(transactions_df,
                         type_name=42,
                         vertex_id_columns=("user_id", "merchant_id"),
                         property_columns=None)
    with pytest.raises(ValueError):
        pG.add_edge_data(transactions_df,
                         type_name="transactions",
                         vertex_id_columns=("user_id", "bad_column"),
                         property_columns=None)
    with pytest.raises(ValueError):
        pG.add_edge_data(transactions_df,
                         type_name="transactions",
                         vertex_id_columns=("user_id", "merchant_id"),
                         property_columns=["bad_column_name", "time"])
    with pytest.raises(TypeError):
        pG.add_edge_data(transactions_df,
                         type_name="transactions",
                         vertex_id_columns=("user_id", "merchant_id"),
                         property_columns="time")


def test_extract_subgraph_vertex_prop_condition_only(property_graph_instance):

    pG = property_graph_instance
    diGraph = cugraph.Graph(directed=True)

    # FIXME: test for proper operators, etc.
    # FIXME: need full PropertyColumn test suite
    vert_prop_cond = "(__type__=='users') & (user_location==78757)"
    G = pG.extract_subgraph(vertex_property_condition=vert_prop_cond,
                            create_using=diGraph,
                            edge_weight_property="relationship_type")

    # FIXME: figure out correct dtypes
    expected_edgelist = cudf.DataFrame({"src": [89216.], "dst": [89021.],
                                        "weights": [9]})
    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src",
                                   preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst",
                                   preserve_order=True)

    assert G.is_directed()
    # check_like=True ignores differences in column/index ordering
    assert_frame_equal(expected_edgelist, actual_edgelist, check_like=True)


def test_extract_subgraph_vertex_edge_prop_condition(property_graph_instance):
    pG = property_graph_instance
    diGraph = cugraph.Graph(directed=True)

    vert_prop_cond = "((user_location==78750) | (user_location==78757))"
    edge_prop_cond = "__type__=='referrals'"
    G = pG.extract_subgraph(vertex_property_condition=vert_prop_cond,
                            edge_property_condition=edge_prop_cond,
                            create_using=diGraph,
                            edge_weight_property="stars")

    expected_edgelist = cudf.DataFrame({"src": [32431.], "dst": [89216.],
                                        "weights": [4]})
    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src",
                                   preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst",
                                   preserve_order=True)

    assert G.is_directed()
    assert_frame_equal(expected_edgelist, actual_edgelist, check_like=True)


def test_extract_subgraph_edge_prop_condition_only(property_graph_instance):
    pG = property_graph_instance
    diGraph = cugraph.Graph(directed=True)

    G = pG.extract_subgraph(edge_property_condition="__type__=='transactions'",
                            create_using=diGraph)

    # last item is the DataFrame rows
    transactions = dataset1["transactions"][-1]
    (srcs, dsts) = zip(*[(float(t[0]), float(t[1])) for t in transactions])
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


def test_extract_subgraph_unweighted(property_graph_instance):
    """
    Ensure a subgraph is unweighted if the edge_weight_property is None.
    """
    pG = property_graph_instance
    diGraph = cugraph.Graph(directed=True)

    G = pG.extract_subgraph(edge_property_condition="__type__=='transactions'",
                            create_using=diGraph)

    assert G.is_weighted() is False


def test_extract_subgraph_specific_query(property_graph_instance):
    """
    Graph of only transactions after time 1639085000 for merchant_id 4 (should
    be a graph of 2 vertices, 1 edge)
    """
    pG = property_graph_instance
    diGraph = cugraph.Graph(directed=True)

    edge_prop_cond = ("(__type__=='transactions') & (merchant_id==4) "
                      "& (time>1639085000)")
    G = pG.extract_subgraph(edge_property_condition=edge_prop_cond,
                            create_using=diGraph,
                            edge_weight_property="volume")

    expected_edgelist = cudf.DataFrame({"src": [89216.], "dst": [4.],
                                        "weights": [12.8]})
    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src",
                                   preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst",
                                   preserve_order=True)

    assert G.is_directed()
    assert_frame_equal(expected_edgelist, actual_edgelist, check_like=True)


def test_extract_subgraph_no_edges(property_graph_instance):
    """
    Valid query that only matches a single vertex.
    """
    pG = property_graph_instance

    vert_prop_cond = "(__type__=='merchants') & (merchant_id==86)"
    G = pG.extract_subgraph(vertex_property_condition=vert_prop_cond)

    assert len(G.edgelist.edgelist_df) == 0


def test_extract_subgraph_no_query(property_graph_instance):
    """
    Call extract with no args, should result in the entire property graph.
    """
    pG = property_graph_instance
    diGraph = cugraph.Graph(directed=True)

    G = pG.extract_subgraph(create_using=diGraph)

    num_edges = len(dataset1["transactions"][-1]) + \
                len(dataset1["relationships"][-1]) + \
                len(dataset1["referrals"][-1])
    # referrals and relationships have an edge in common, so subtract it from
    # the total count.
    num_edges -= 1
    assert len(G.edgelist.edgelist_df) == num_edges


def test_extract_subgraph_bad_args(property_graph_instance):
    pG = property_graph_instance
    diGraph = cugraph.Graph(directed=True)

    # non-string condition
    with pytest.raises(TypeError):
        pG.extract_subgraph(vertex_property_condition=78750,
                            create_using=diGraph,
                            edge_weight_property="stars",
                            default_edge_weight=1.0)
    # bad create_using type
    with pytest.raises(TypeError):
        pG.extract_subgraph(edge_property_condition="__type__=='referrals'",
                            create_using=pytest,
                            edge_weight_property="stars",
                            default_edge_weight=1.0)
    # invalid column name
    with pytest.raises(ValueError):
        pG.extract_subgraph(edge_property_condition="__type__=='referrals'",
                            edge_weight_property="bad_column",
                            default_edge_weight=1.0)
    # column name has None value for all results in subgraph and
    # default_edge_weight is not set.
    with pytest.raises(ValueError):
        pG.extract_subgraph(edge_property_condition="__type__=='referrals'",
                            edge_weight_property="card_type")


@pytest.mark.skip(reason="unfinished")
def test_property_lookup():
    pass


def test_extract_subgraph_default_edge_weight(property_graph_instance):
    pG = property_graph_instance
    diGraph = cugraph.Graph(directed=True)

    G = pG.extract_subgraph(create_using=diGraph,
                            edge_property_condition="__type__=='transactions'",
                            edge_weight_property="card_num",
                            default_edge_weight=99)

    # last item is the DataFrame rows
    transactions = dataset1["transactions"][-1]
    (srcs, dsts, weights) = zip(*[(float(t[0]), float(t[1]), t[4])
                                  for t in transactions])
    # replace None with the expected value (convert to a list to replace)
    weights_list = list(weights)
    weights_list[weights.index(None)] = 99
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


@pytest.mark.skip(reason="unfinished")
def test_property_edge_id():
    pass
