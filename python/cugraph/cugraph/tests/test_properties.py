# Copyright (c) 2021, NVIDIA CORPORATION.
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

import pytest

import pandas as pd
import cudf
from cudf.testing import assert_frame_equal


dataset1 = {
    "merchants" : [
        ["merchant_id", "merchant_location", "merchant_size", "merchant_sales", "merchant_num_employees", "merchant_name"],
        [(11, 78750, 44, 123.2, 12, "north"),
         (4, 78757, 112, 234.99, 18, "south"),
         (21, 44145, 83, 992.1, 27, "east"),
         (16, 47906, 92, 32.43, 5, "west"),
        ]
    ],
    "users" : [
        ["user_id", "user_location", "vertical"],
        [(89021, 78757, 0),
         (32431, 78750, 1),
         (89216, 78757, 1),
         (78634, 47906, 0),
        ]
    ],
    "transactions" : [
        ["user_id", "merchant_id", "volume", "time", "card_num", "card_type"],
        [(89021, 11, 33.2, 1639084966.5513437, 123456, "MC"),
         (89216, 4, 12.8, 1639085163.481217, None, "CASH"),
         (78634, 16, 72.0, 1639084912.567394, 4321, "DEBIT"),
         (32431, 4, 103.2, 1639084721.354346, 98124, "V"),
        ]
    ],
    "relationships" : [
        ["user_id_1", "user_id_2", "relationship_type"],
        [(89216, 89021, 0),
         (89216, 32431, 0),
         (32431, 78634, 1),
         (78634, 89216, 1),
        ]
    ],
    "referrals" : [
        ["user_id_1", "user_id_2", "merchant_id", "stars"],
        [(89216, 78634, 11, 5),
         (32431, 89216, 4, 4),
         (89021, 78634, 21, 4),
         (78634, 89216, 11, 4),
        ]
    ],
}


################################################################################
## Tests

def test_add_vertex_data():
    """
    add_vertex_data() on "merchants" table, all properties.
    """
    from cugraph import PropertyGraph

    merchants = dataset1["merchants"]
    merchants_df = cudf.DataFrame(columns=merchants[0],
                                  data=merchants[1])

    pG = PropertyGraph()
    pG.add_vertex_data(merchants_df,
                       type_name="merchants",
                       vertex_id_column="merchant_id",
                       property_columns=None)

    assert pG.num_vertices == 4
    assert pG.num_edges == 0
    expected_props = merchants[0].copy()
    expected_props.remove("merchant_id")
    assert sorted(pG.vertex_properties) == sorted(expected_props)


def test_add_vertex_data_prop_columns():
    """
    add_vertex_data() on "merchants" table, subset of properties.
    """
    from cugraph import PropertyGraph

    merchants = dataset1["merchants"]
    merchants_df = cudf.DataFrame(columns=merchants[0],
                                  data=merchants[1])
    expected_props = ["merchant_name", "merchant_sales", "merchant_location"]

    pG = PropertyGraph()
    pG.add_vertex_data(merchants_df,
                       type_name="merchants",
                       vertex_id_column="merchant_id",
                       property_columns=expected_props)

    assert pG.num_vertices == 4
    assert pG.num_edges == 0
    assert sorted(pG.vertex_properties) == sorted(expected_props)


def test_add_vertex_data_bad_args():
    """
    add_vertex_data() with various bad args, checks that proper exceptions are raised.
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
                           property_columns=["bad_column_name", "merchant_name"])
    with pytest.raises(TypeError):
        pG.add_vertex_data(merchants_df,
                           type_name="merchants",
                           vertex_id_column="merchant_id",
                           property_columns="merchant_name")


def test_add_edge_data():
    """
    add_edge_data() on "transactions" table, all properties.
    """
    from cugraph import PropertyGraph

    transactions = dataset1["transactions"]
    transactions_df = cudf.DataFrame(columns=transactions[0],
                                     data=transactions[1])

    pG = PropertyGraph()
    pG.add_edge_data(transactions_df,
                     type_name="transactions",
                     vertex_id_columns=("user_id", "merchant_id"),
                     property_columns=None)

    assert pG.num_vertices == 7
    assert pG.num_edges == 4
    expected_props = ["volume", "time", "card_num", "card_type"]
    assert sorted(pG.edge_properties) == sorted(expected_props)


def test_add_edge_data_prop_columns():
    """
    add_edge_data() on "transactions" table, subset of properties.
    """
    from cugraph import PropertyGraph

    transactions = dataset1["transactions"]
    transactions_df = cudf.DataFrame(columns=transactions[0],
                                     data=transactions[1])
    expected_props = ["card_num", "card_type"]

    pG = PropertyGraph()
    pG.add_edge_data(transactions_df,
                     type_name="transactions",
                     vertex_id_columns=("user_id", "merchant_id"),
                     property_columns=expected_props)

    assert pG.num_vertices == 7
    assert pG.num_edges == 4
    assert sorted(pG.edge_properties) == sorted(expected_props)


def test_add_edge_data_bad_args():
    """
    add_edge_data() with various bad args, checks that proper exceptions are raised.
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


def test_PropertyGraph_complex_queries():

    from cugraph import Graph, PropertyGraph

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
    pG.add_vertex_data(cudf.DataFrame(columns=merchants[0],
                                      data=merchants[1]),
                       type_name="merchants",
                       vertex_id_column="merchant_id",
                       property_columns=None)
    pG.add_vertex_data(cudf.DataFrame(columns=users[0],
                                      data=users[1]),
                       type_name="users",
                       vertex_id_column="user_id",
                       property_columns=None)

    pG.add_edge_data(cudf.DataFrame(columns=transactions[0],
                                    data=transactions[1]),
                     type_name="transactions",
                     vertex_id_columns=("user_id", "merchant_id"),
                     property_columns=None)
    pG.add_edge_data(cudf.DataFrame(columns=relationships[0],
                                    data=relationships[1]),
                     type_name="relationships",
                     vertex_id_columns=("user_id_1", "user_id_2"),
                     property_columns=None)
    pG.add_edge_data(cudf.DataFrame(columns=referrals[0],
                                    data=referrals[1]),
                     type_name="referrals",
                     vertex_id_columns=("user_id_1",
                                        "user_id_2"),
                     property_columns=None)

    # Graph of only relationship_type 1
    diGraph = Graph(directed=True)
    # FIXME: test for proper operators, etc. Need full PropertyColumn test suite
    G = pG.extract_subgraph(vertex_property_condition="(type_name=='users') & (user_location==78757)",
                            create_using=diGraph,
                            edge_weight_property="relationship_type",
                            default_edge_weight=1.0)

    # FIXME: figure out correct dtypes
    expected_edgelist = cudf.DataFrame({"src":[89216.], "dst":[89021.]})
    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src", preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst", preserve_order=True)

    assert G.is_directed()
    # FIXME: need to test for edge IDs
    assert_frame_equal(expected_edgelist, actual_edgelist)

    ########
    G = pG.extract_subgraph(vertex_property_condition="((user_location==78750) | (user_location==78757))",
                            edge_property_condition="type_name=='referrals'",
                            create_using=diGraph,
                            edge_weight_property="stars",
                            default_edge_weight=1.0)

    expected_edgelist = cudf.DataFrame({"src":[32431.], "dst":[89216.]})
    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src", preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst", preserve_order=True)

    assert G.is_directed()
    # FIXME: need to test for edge IDs
    assert_frame_equal(expected_edgelist, actual_edgelist)

    # Graph of only transactions
    # Graph of only users
    # Graph of only merchants (should result in no edges)
    # Graph of only transactions after time 1639085000 for merchant_id 4 (should be a graph of 2 vertices, 1 edge)
