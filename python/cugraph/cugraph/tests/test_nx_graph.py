import pytest
import pandas as pd
import cudf
import cugraph as cnx
from cugraph.nx.Graph import Graph
from cugraph.tests import utils

dataset = {
    "newedges": [
    ["source", "destination"],
        [(1, 2),
         (2, 3),
         (4, 2),
         (3, 5),
         (6, 2),
         ]
     ],}


def test_add_node():
    cnG = Graph()
    cnG.add_node(4)
    assert cnG.number_of_nodes()  == 1
    cnG.add_edge(1, 4)
    assert cnG.number_of_edges() == 1
    assert cnG.number_of_nodes() == 2 
    cnG.add_edges_from([(1, 2), (1, 3)])
    assert cnG.number_of_edges() == 3
    assert cnG.number_of_nodes() == 4


# Pytest fixtures
# =============================================================================
# =============================================================================
df_types = [cudf.DataFrame, pd.DataFrame]


def df_type_id(dft):
    s = "df_type="
    if dft == cudf.DataFrame:
        return s+"cudf.DataFrame"
    return s+"?"


@pytest.fixture(scope="module",
                params=utils.genFixtureParamsProduct((df_types, df_type_id))
                )
def property_graph_instance(request):
    """
    FIXME: fill this in
    """
    dataframe_type = request.param[0]
    from cugraph.experimental import PropertyGraph

    pG = PropertyGraph()
    (newedges) = dataset.values()

    pG.add_vertex_data(dataframe_type(columns=newedges[0],
                                      data=newedges[1]),
                       type_name="node_type",
                       vertex_id_columns=("source", "destination"),
                       property_columns=None)

    return pG


###############################################################################
# Tests
@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_add_vertex_data(df_type):
    """
    add_edge_data() on "merchants" table, all properties.
    """
    from cugraph.experimental import PropertyGraph

    newedges = dataset["newedges"]
    newedges_df = df_type(columns=newedges[0],
                           data=newedges[1])


    pG = PropertyGraph()
    
    pG.add_edge_data(newedges_df,
    vertex_id_columns=("source", "destination"), 
        type_name="newedges", 
        property_columns=None)
    assert pG.num_vertices == 6
    assert pG.num_edges == 5


def test_nx_property_edges():

    cnG = Graph()

    newedges = dataset["newedges"]
    newedges_df = cudf.DataFrame(columns=newedges[0], data=newedges[1])
    cnG.add_edges_from(newedges_df)

    assert cnG.number_of_nodes() == 6
    assert cnG.number_of_edges() == 5


def test_nx_add_node_s():
    cnG = Graph()
    cnG.add_node("newnode1")
    assert cnG.number_of_nodes() == 1
    cnG.add_node("newnode2")
    cnG.add_node("newnode3")
    assert cnG.number_of_nodes() == 3
    cnG.add_node("newnode3")
    assert cnG.number_of_nodes() == 3


def test_nx_add_edge():
    cnG = Graph()
    cnG.add_edge("newnode1", "newnode2")
    assert cnG.number_of_nodes() == 2
    assert cnG.number_of_edges() == 1
    cnG.add_edge("newnode1", "newnode3")
    assert cnG.number_of_nodes() == 3
    assert cnG.number_of_edges() == 2
    cnG.add_edge("newnode2", "newnode3")
    assert cnG.number_of_nodes() == 3
    assert cnG.number_of_edges() == 3
    cnG.add_edge("newnode2", "newnode3")
    assert cnG.number_of_nodes() == 3
    assert cnG.number_of_edges() == 3

def test_nx_add_edge_from_list():
    # Add some edges
    G = Graph()
    G.add_edges_from([(1,2), (1, 3)])
    assert G.number_of_edges() == 2
    assert G.number_of_nodes() == 3


def test_nx_edges():
    testdata = {"edgedata": [
        ["user_id_1", "user_id_2", "edge_type"],
        [(1, 2, "type1"),
         (1, 3, "type2"),
         (3, 4, "type1"),
         (4, 2, "type2"),
         ]]}
    edges = testdata["edgedata"]
    cnG = Graph()
    cnG.add_edges_from(edges)
    assert cnG.number_of_edges() == 4
    assert cnG.number_of_nodes() == 4

def test_bfg():
    G = Graph()
    G.add_edges_from([(1, 2), (1, 3)])
    G.add_node("spam")       # adds node "spam"
    cG = G.as_cugraph()
    #assert list(cnx.connected_components(cG)) == [{1, 2, 3}, {'spam'}]
    #assert sorted(d for n, d in cG.degree()) == [0, 1, 1, 2]
    #assert cnx.clustering(cG) == {1: 0, 2: 0, 3: 0, 'spam': 0}
    #assert list(cnx.bfs_edges(cG, 1)) == [(1, 2), (1, 3)]

