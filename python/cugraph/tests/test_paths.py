import cudf
import cugraph
from cupy.sparse import coo_matrix as cupy_coo_matrix
import cupy
import networkx as nx
import pytest
import sys
from tempfile import NamedTemporaryFile

CONNECTED_GRAPH = """1,5,3
1,4,1
1,2,1
1,6,2
1,7,2
4,5,1
2,3,1
7,6,2
"""

DISCONNECTED_GRAPH = CONNECTED_GRAPH + "8,9,4"


@pytest.fixture
def graphs(request):
    with NamedTemporaryFile(mode="w+", suffix=".csv") as graph_tf:
        graph_tf.writelines(request.param)
        graph_tf.seek(0)

        nx_G = nx.read_weighted_edgelist(graph_tf.name, delimiter=',')
        cudf_df = cudf.read_csv(graph_tf.name,
                                names=["src", "dst", "data"],
                                delimiter=",",
                                dtype=["int32", "int32", "float64"])
        cugraph_G = cugraph.Graph()
        cugraph_G.from_cudf_edgelist(
                                    cudf_df, source="src",
                                    destination="dst", edge_attr="data")

        # construct cupy coo_matrix graph
        i = []
        j = []
        weights = []
        for index in range(cudf_df.shape[0]):
            vertex1 = cudf_df.iloc[index]["src"]
            vertex2 = cudf_df.iloc[index]["dst"]
            weight = cudf_df.iloc[index]["data"]
            i += [vertex1, vertex2]
            j += [vertex2, vertex1]
            weights += [weight, weight]
        i = cupy.array(i)
        j = cupy.array(j)
        weights = cupy.array(weights)
        largest_vertex = max(cupy.amax(i), cupy.amax(j))
        cupy_df = cupy_coo_matrix(
            (weights, (i, j)),
            shape=(largest_vertex + 1, largest_vertex + 1))

        yield cugraph_G, nx_G, cupy_df


@pytest.mark.parametrize("graphs", [CONNECTED_GRAPH], indirect=True)
def test_connected_graph_shortest_path_length(graphs):
    cugraph_G, nx_G, cupy_df = graphs

    path_1_to_1_length = cugraph.shortest_path_length(cugraph_G, 1, 1)
    assert path_1_to_1_length == 0.0
    assert path_1_to_1_length == nx.shortest_path_length(
        nx_G, "1", target="1", weight="weight")
    assert path_1_to_1_length == cugraph.shortest_path_length(nx_G, "1", "1")
    assert path_1_to_1_length == cugraph.shortest_path_length(cupy_df, 1, 1)

    path_1_to_5_length = cugraph.shortest_path_length(cugraph_G, 1, 5)
    assert path_1_to_5_length == 2.0
    assert path_1_to_5_length == nx.shortest_path_length(
        nx_G, "1", target="5", weight="weight")
    assert path_1_to_5_length == cugraph.shortest_path_length(nx_G, "1", "5")
    assert path_1_to_5_length == cugraph.shortest_path_length(cupy_df, 1, 5)

    path_1_to_3_length = cugraph.shortest_path_length(cugraph_G, 1, 3)
    assert path_1_to_3_length == 2.0
    assert path_1_to_3_length == nx.shortest_path_length(
        nx_G, "1", target="3", weight="weight")
    assert path_1_to_3_length == cugraph.shortest_path_length(nx_G, "1", "3")
    assert path_1_to_3_length == cugraph.shortest_path_length(cupy_df, 1, 3)

    path_1_to_6_length = cugraph.shortest_path_length(cugraph_G, 1, 6)
    assert path_1_to_6_length == 2.0
    assert path_1_to_6_length == nx.shortest_path_length(
        nx_G, "1", target="6", weight="weight")
    assert path_1_to_6_length == cugraph.shortest_path_length(nx_G, "1", "6")
    assert path_1_to_6_length == cugraph.shortest_path_length(cupy_df, 1, 6)


@pytest.mark.parametrize("graphs", [CONNECTED_GRAPH], indirect=True)
def test_shortest_path_length_invalid_source(graphs):
    cugraph_G, nx_G, cupy_df = graphs

    with pytest.raises(ValueError):
        cugraph.shortest_path_length(cugraph_G, -1, 1)

    with pytest.raises(ValueError):
        cugraph.shortest_path_length(nx_G, "-1", "1")

    with pytest.raises(ValueError):
        cugraph.shortest_path_length(cupy_df, -1, 1)


@pytest.mark.parametrize("graphs", [DISCONNECTED_GRAPH], indirect=True)
def test_shortest_path_length_invalid_target(graphs):
    cugraph_G, nx_G, cupy_df = graphs

    with pytest.raises(ValueError):
        cugraph.shortest_path_length(cugraph_G, 1, 10)

    with pytest.raises(ValueError):
        cugraph.shortest_path_length(nx_G, "1", "10")

    with pytest.raises(ValueError):
        cugraph.shortest_path_length(cupy_df, 1, 10)


@pytest.mark.parametrize("graphs", [CONNECTED_GRAPH], indirect=True)
def test_shortest_path_length_invalid_vertexes(graphs):
    cugraph_G, nx_G, cupy_df = graphs

    with pytest.raises(ValueError):
        cugraph.shortest_path_length(cugraph_G, 0, 42)

    with pytest.raises(ValueError):
        cugraph.shortest_path_length(nx_G, "0", "42")

    with pytest.raises(ValueError):
        cugraph.shortest_path_length(cupy_df, 0, 42)


@pytest.mark.parametrize("graphs", [DISCONNECTED_GRAPH], indirect=True)
def test_shortest_path_length_no_path(graphs):
    cugraph_G, nx_G, cupy_df = graphs

    path_1_to_8 = cugraph.shortest_path_length(cugraph_G, 1, 8)
    assert path_1_to_8 == sys.float_info.max
    assert path_1_to_8 == cugraph.shortest_path_length(nx_G, "1", "8")
    assert path_1_to_8 == cugraph.shortest_path_length(cupy_df, 1, 8)


@pytest.mark.parametrize("graphs", [DISCONNECTED_GRAPH], indirect=True)
def test_shortest_path_length_no_target(graphs):
    cugraph_G, nx_G, cupy_df = graphs

    cugraph_path_1_to_all = cugraph.shortest_path_length(cugraph_G, 1)
    nx_path_1_to_all = nx.shortest_path_length(
        nx_G, source="1", weight="weight")
    nx_gpu_path_1_to_all = cugraph.shortest_path_length(nx_G, "1")
    cupy_path_1_to_all = cugraph.shortest_path_length(cupy_df, 1)

    # Cast networkx graph on cugraph vertex column type from str to int.
    # SSSP preserves vertex type, convert for comparison
    nx_gpu_path_1_to_all["vertex"] = \
        nx_gpu_path_1_to_all["vertex"].astype("int32")

    assert cugraph_path_1_to_all == nx_gpu_path_1_to_all
    assert cugraph_path_1_to_all == cupy_path_1_to_all

    # results for vertex 8 and 9 are not returned
    assert cugraph_path_1_to_all.shape[0] == len(nx_path_1_to_all) + 2

    for index in range(cugraph_path_1_to_all.shape[0]):

        vertex = str(cugraph_path_1_to_all["vertex"][index].item())
        distance = cugraph_path_1_to_all["distance"][index].item()

        # verify cugraph against networkx
        if vertex in {'8', '9'}:
            # Networkx does not return distances for these vertexes.
            assert distance == sys.float_info.max
        else:
            assert distance == nx_path_1_to_all[vertex]
