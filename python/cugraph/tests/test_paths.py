import cudf
import cugraph
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

        nx_graph = nx.read_weighted_edgelist(graph_tf.name, delimiter=',')
        gpu_df = cudf.read_csv(graph_tf.name, names=["src", "dst", "data"],
                               delimiter=",", dtype=["int32", "int32", "float64"])
        gpu_graph = cugraph.Graph()
        gpu_graph.from_cudf_edgelist(gpu_df, source="src", destination="dst", edge_attr="data")

        yield gpu_graph, nx_graph


@pytest.mark.parametrize("graphs", [CONNECTED_GRAPH], indirect=True)
def test_connected_graph_shortest_path_length(graphs):
    gpu_graph, nx_graph = graphs

    path_1_to_1_length = cugraph.shortest_path_length(gpu_graph, 1, 1)
    assert path_1_to_1_length == 0.0
    assert path_1_to_1_length == nx.shortest_path_length(nx_graph, "1", target="1", weight="weight")

    path_1_to_5_length = cugraph.shortest_path_length(gpu_graph, 1, 5)
    assert path_1_to_5_length == 2.0
    assert path_1_to_5_length == nx.shortest_path_length(nx_graph, "1", target="5", weight="weight")

    path_1_to_3_length = cugraph.shortest_path_length(gpu_graph, 1, 3)
    assert path_1_to_3_length == 2.0
    assert path_1_to_3_length == nx.shortest_path_length(nx_graph, "1", target="3", weight="weight")

    path_1_to_6_length = cugraph.shortest_path_length(gpu_graph, 1, 6)
    assert path_1_to_6_length == 2.0
    assert path_1_to_6_length == nx.shortest_path_length(nx_graph, "1", target="6", weight="weight")


@pytest.mark.parametrize("graphs", [CONNECTED_GRAPH], indirect=True)
def test_shortest_path_length_invalid_source(graphs):
    gpu_graph, __ = graphs

    with pytest.raises(ValueError):
        cugraph.shortest_path_length(gpu_graph, 42, 1)


@pytest.mark.parametrize("graphs", [CONNECTED_GRAPH], indirect=True)
def test_shortest_path_length_invalid_target(graphs):
    gpu_graph, __ = graphs

    with pytest.raises(ValueError):
        cugraph.shortest_path_length(gpu_graph, 1, 42)


@pytest.mark.parametrize("graphs", [CONNECTED_GRAPH], indirect=True)
def test_shortest_path_length_invalid_vertexes(graphs):
    gpu_graph, __ = graphs

    with pytest.raises(ValueError):
        cugraph.shortest_path_length(gpu_graph, 42, 42)


@pytest.mark.parametrize("graphs", [DISCONNECTED_GRAPH], indirect=True)
def test_shortest_path_length_no_path(graphs):
    gpu_graph, __ = graphs

    path_1_to_8 = cugraph.shortest_path_length(gpu_graph, 1, 8)
    assert path_1_to_8 == sys.float_info.max


@pytest.mark.parametrize("graphs", [DISCONNECTED_GRAPH], indirect=True)
def test_shortest_path_length_no_target(graphs):
    gpu_graph, nx_graph = graphs

    gpu_path_1_to_all = cugraph.shortest_path_length(gpu_graph, 1)
    nx_path_1_to_all = nx.shortest_path_length(nx_graph, source="1", weight="weight")
    assert gpu_path_1_to_all.shape[0] == len(nx_path_1_to_all) + 2

    for index in range(gpu_path_1_to_all.shape[0]):

        vertex = str(gpu_path_1_to_all["vertex"][index].item())
        distance = gpu_path_1_to_all["distance"][index].item()

        if vertex in {'8', '9'}:
            assert distance == sys.float_info.max
        else:
            assert distance == nx_path_1_to_all[vertex]
