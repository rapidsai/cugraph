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
    "merchants": [
        ["merchant_location", "merchant_id", "merchant_size"],
        [(78750, 11, 44),
         (78757, 4, 112),
         (44145, 21, 83),
         (47906, 16, 92),
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
        ["user_id", "merchant_id", "volume", "time"],
        [(89021, 11, 33.2, 1639084966.5513437),
         (89216, 4, 12.8, 1639085163.481217),
         (78634, 16, 72.0, 1639084912.567394),
         (32431, 4, 103.2, 1639084721.354346),
        ]
    ],
    "relationships": [
        ["user_id_1", "user_id_2", "relationship_type"],
        [(89216, 89021, 0),
         (89216, 32431, 0),
         (32431, 78634, 1),
         (78634, 89216, 1),
        ]
    ],
    "personal_loans": [
        ["lender_user_id", "borrower_user_id", "amount"],
        [(89216, 78634, 11.23),
         (32431, 89216, 9.32),
         (89021, 78634, 10.21),
         (78634, 89216, 2.87),
        ]
    ],
}


def test_PropertyGraph_complex_queries():

    import cugraph

    (merchants, users,
     transactions, relationships, personal_loans) = dataset1.values()

    pG = cugraph.PropertyGraph()

    # Vertex and edge data is added as one or more DataFrames; either a Pandas
    # DataFrame to keep data on the CPU, a cuDF DataFrame to keep data on GPU,
    # or a dask_cudf DataFrame to keep data on distributed GPUs.

    # For dataset1: vertices are merchants and users, edges are transactions,
    # relationships, and personal_loans.

    # property_columns=None (the default) means all columns except
    # vertex_id_column will be used as properties for the vertices/edges.
    pG.add_vertex_data(pd.DataFrame(columns=merchants[0],
                                    data=merchants[1]),
                       type_name="merchants",
                       vertex_id_column="merchant_id",
                       property_columns=None)
    pG.add_vertex_data(pd.DataFrame(columns=users[0],
                                    data=users[1]),
                       type_name="users",
                       vertex_id_column="user_id",
                       property_columns=None)

    pG.add_edge_data(pd.DataFrame(columns=transactions[0],
                                  data=transactions[1]),
                     type_name="transactions",
                     edge_vertices_columns=("user_id", "merchant_id"),
                     property_columns=None)
    pG.add_edge_data(pd.DataFrame(columns=relationships[0],
                                  data=relationships[1]),
                     type_name="relationships",
                     edge_vertices_columns=("user_id_1", "user_id_2"),
                     property_columns=None)
    pG.add_edge_data(pd.DataFrame(columns=personal_loans[0],
                                  data=personal_loans[1]),
                     type_name="personal_loans",
                     edge_vertices_columns=("lender_user_id",
                                            "borrower_user_id"),
                     property_columns=None)

    # Graph of only relationship_type 1
    diGraph = cugraph.Graph(directed=True)
    # FIXME: test for proper operators, etc. Need full PropertyColumn test suite
    G = pG.extract_subgraph(vertex_property_condition="(type_name=='users') & (user_location==78757)",
                            create_using=diGraph)

    expected_edgelist = cudf.DataFrame({"src":[89216], "dst":[89021]})
    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src", preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst", preserve_order=True)

    assert G.is_directed()
    # FIXME: need to test for edge IDs
    assert_frame_equal(expected_edgelist, actual_edgelist)

    ########
    G = pG.extract_subgraph(vertex_property_condition="((user_location==78750) | (user_location==78757))",
                            edge_property_condition="type_name=='personal_loans'",
                            create_using=diGraph)

    expected_edgelist = cudf.DataFrame({"src":[32431], "dst":[89216]})
    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src", preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst", preserve_order=True)

    assert G.is_directed()
    # FIXME: need to test for edge IDs
    assert_frame_equal(expected_edgelist, actual_edgelist)

    # Graph of only transactions
    # Graph of only users
    # Graph of only merchants (should result in no edges)
    # Graph of only transactions after time 1639085000 for merchant_id 4 (should be a graph of 2 vertices, 1 edge)



@pytest.mark.skip(reason="incomplete test")
def test_add_vertex_data():
    (merchants, users, transactions, relationships) = dataset1.values()

    pG = cugraph.PropertyGraph()

    # Invalid column name
    # Missing vertex_id column name
    # Single column name
    # Bad DataFrame
    pG.add_vertex_data(pd.DataFrame(columns=users[0],
                                    data=users[1]),
                       type_name="users",
                       vertex_id_column="user_id",
                       property_columns=None)


@pytest.mark.skip(reason="incomplete test")
def test_add_edge_data():
    (merchants, users, transactions, relationships) = dataset1.values()

    pG = cugraph.PropertyGraph()

    # Invalid column name
    # Missing src/dst column name
    # Single column name
    # Bad DataFrame
    pG.add_edge_data(pd.DataFrame(columns=transactions[0],
                                  data=transactions[1]),
                     type_name="transactions",
                     edge_vertices_columns=("user_id", "merchant_id"),
                     property_columns=None)









"""
    # Create the graph edgelist: this is done by extracting specific columns
    # from the "transactions" and "relationships" tables and combining them.
    # In the case of "transactions" and "relationships", the first two columns of each
    # describe all edges in the graph.
    edgelist_df = cudf.DataFrame(columns=["src", "dst"],
                                 data=[row[0:2] for row in
                                       transactions[1] + relationships[1]]
                                 )

    # Add vertex properties: "merchants" and "users" tables. Vertex "type"
    # (merchant or user) is also added.
    vertex_props_df = cudf.DataFrame()

    # Add edge properties: "transactions" and "relationships" tables. Edge "type"
    # (transaction or relationships) is also added.
    edge_props_df = cudf.DataFrame()


empty_DiGraph = cugraph.Graph(directed=True)

G = cugraph.from_cudf_edgelist(...,
                               create_using=empty_DiGraph)
"""
