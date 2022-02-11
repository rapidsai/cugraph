import pytest
import pandas as pd
import cudf
import cugraph
from cugraph.nx.Graph import Graph
from cugraph.tests import utils

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




###############################################################################
# Tests
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


@pytest.mark.parametrize("df_type", df_types)
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


def nx_edges():
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

def bfs():
    G = Graph()
    G.add_edges_from([(1, 2), (1, 3)])
    G.add_node("spam")       # adds node "spam"
    cG = G.as_cugraph()
    # print (cugraph.connected_components(cG))
    # print ( sorted(d for n, d in cG.degree()))
    # print ( cugraph.clustering(cG))
    # print ( cugraph.bfs_edges(cG, 1))


@pytest.mark.parametrize(
    "graph_file",
    [utils.RAPIDS_DATASET_ROOT_DIR_PATH/"dolphins.csv"])
def with_dolphins(graph_file):

    import networkx as nx

    df = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)
    G = nx.from_pandas_edgelist(df, create_using=nx.Graph(),
                                source="0", target="1", edge_attr="weight") 
    assert G.degree(0) == 6
    assert G.degree(14) == 12
    assert G.degree(15) == 7
    assert G.degree(40) == 8
    assert G.degree(42) == 6
    assert G.degree(47) == 6
    assert G.degree(17) == 9


@pytest.mark.parametrize(
    "graph_file",
    [utils.RAPIDS_DATASET_ROOT_DIR_PATH/"dolphins.csv"])
def bench_build_dolphins(gpubenchmark, graph_file):
    import networkx as nx

    def func():
        df = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)
        gpuG = nx.from_pandas_edgelist(df, create_using=nx.Graph(),
                                       source="0", target="1", edge_attr="weight")


    gpubenchmark(func)


@pytest.mark.parametrize(
    "graph_file",
    [utils.RAPIDS_DATASET_ROOT_DIR_PATH/"dolphins.csv"])
def bench_build_nx_dolphins(gpubenchmark,graph_file):

    def func():
        df = cudf.read_csv(graph_file, names=["src", "dst"],
                           delimiter='\t', dtype=["int32", "int32"] )
        nG = Graph()
        nG.add_edges_from(df)


    gpubenchmark(func)


@pytest.mark.parametrize(
    "graph_file",
    [utils.RAPIDS_DATASET_ROOT_DIR_PATH/"cyber.csv"])
def bench_build_cyber(gpubenchmark,graph_file):
    import networkx as nx

    source_col_name = "srcip"
    dest_col_name = "dstip"

    def func():
        df = pd.read_csv(graph_file, delimiter=",",dtype={"idx": "int32",
                         source_col_name: "str",
                         dest_col_name: "str"},
                         header=0
                         )
        gpuG = nx.from_pandas_edgelist(df,
                                       create_using=nx.Graph(),
                                       source=source_col_name, target=dest_col_name)
    gpubenchmark(func)


@pytest.mark.parametrize(
    "graph_file",
    [utils.RAPIDS_DATASET_ROOT_DIR_PATH/"cyber.csv"])
def bench_build_nx_cyber(gpubenchmark, graph_file):
 
    source_col_name = "srcip"
    dest_col_name = "dstip"

    def func():
        df = cudf.read_csv(graph_file, delimiter=",",
                  dtype={"idx": "int32",
                         source_col_name: "str",
                         dest_col_name: "str"},
                  header=0)
        nG = Graph()
        nG.add_edges_from(df)


    gpubenchmark(func)
