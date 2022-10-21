# Copyright (c) 2022, NVIDIA CORPORATION.
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

from collections.abc import Sequence

import pytest

from . import data
from . import utils


###############################################################################
# fixtures


@pytest.fixture(scope="module")
def server():
    """
    Start a cugraph_service server, stop it when done with the fixture.
    """
    from cugraph_service_client import CugraphServiceClient
    from cugraph_service_client.exceptions import CugraphServiceError

    host = "localhost"
    port = 9090
    client = CugraphServiceClient(host, port)

    try:
        client.uptime()
        print("FOUND RUNNING SERVER, ASSUMING IT SHOULD BE USED FOR TESTING!")
        yield

    except CugraphServiceError:
        # A server was not found, so start one for testing then stop it when
        # testing is done.
        server_process = utils.start_server_subprocess(host=host, port=port)

        # yield control to the tests, cleanup on return
        yield

        # tests are done, now stop the server
        print("\nTerminating server...", end="", flush=True)
        server_process.terminate()
        print("done.", flush=True)


@pytest.fixture(scope="function")
def client(server):
    """
    Creates a client instance to the running server, closes the client when the
    fixture is no longer used by tests.
    """
    from cugraph_service_client import CugraphServiceClient, defaults

    client = CugraphServiceClient(defaults.host, defaults.port)

    for gid in client.get_graph_ids():
        client.delete_graph(gid)

    # FIXME: should this fixture always unconditionally unload all extensions?
    # client.unload_graph_creation_extensions()

    # yield control to the tests, cleanup on return
    yield client

    client.close()


@pytest.fixture(scope="function")
def client_with_graph_creation_extension_loaded(client, graph_creation_extension1):
    """
    Loads the extension defined in graph_creation_extension1, unloads upon completion.
    """
    server_extension_dir = graph_creation_extension1

    client.load_graph_creation_extensions(server_extension_dir.name)

    # yield control to the tests, cleanup on return
    yield client

    client.unload_graph_creation_extensions()


@pytest.fixture(scope="function")
def client_with_edgelist_csv_loaded(client):
    """
    Loads the karate CSV into the default graph on the server.
    """
    test_data = data.edgelist_csv_data["karate"]
    client.load_csv_as_edge_data(
        test_data["csv_file_name"],
        dtypes=test_data["dtypes"],
        vertex_col_names=["0", "1"],
        type_name="",
    )
    assert client.get_graph_ids() == [0]
    return (client, test_data)


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
    return (client, data.property_csv_data)


###############################################################################
# tests
def test_get_graph_info_key_types(client_with_property_csvs_loaded):
    """
    Tests error handling for info keys passed in.
    """
    from cugraph_service_client.exceptions import CugraphServiceError

    (client, test_data) = client_with_property_csvs_loaded

    with pytest.raises(TypeError):
        client.get_graph_info(21)  # bad key type
    with pytest.raises(TypeError):
        client.get_graph_info([21, "num_edges"])  # bad key type
    with pytest.raises(CugraphServiceError):
        client.get_graph_info("21")  # bad key value
    with pytest.raises(CugraphServiceError):
        client.get_graph_info(["21"])  # bad key value
    with pytest.raises(CugraphServiceError):
        client.get_graph_info(["num_edges", "21"])  # bad key value

    client.get_graph_info()  # valid


def test_get_num_edges_default_graph(client_with_edgelist_csv_loaded):
    (client, test_data) = client_with_edgelist_csv_loaded
    assert client.get_graph_info("num_edges") == test_data["num_edges"]


def test_load_csv_as_edge_data_nondefault_graph(client):
    from cugraph_service_client.exceptions import CugraphServiceError

    test_data = data.edgelist_csv_data["karate"]

    with pytest.raises(CugraphServiceError):
        client.load_csv_as_edge_data(
            test_data["csv_file_name"],
            dtypes=test_data["dtypes"],
            vertex_col_names=["0", "1"],
            type_name="",
            graph_id=9999,
        )


def test_get_num_edges_nondefault_graph(client_with_edgelist_csv_loaded):
    from cugraph_service_client.exceptions import CugraphServiceError

    (client, test_data) = client_with_edgelist_csv_loaded
    # Bad graph ID
    with pytest.raises(CugraphServiceError):
        client.get_graph_info("num_edges", graph_id=9999)

    new_graph_id = client.create_graph()
    client.load_csv_as_edge_data(
        test_data["csv_file_name"],
        dtypes=test_data["dtypes"],
        vertex_col_names=["0", "1"],
        type_name="",
        graph_id=new_graph_id,
    )

    assert client.get_graph_info("num_edges") == test_data["num_edges"]
    assert (
        client.get_graph_info("num_edges", graph_id=new_graph_id)
        == test_data["num_edges"]
    )


def test_node2vec(client_with_edgelist_csv_loaded):
    (client, test_data) = client_with_edgelist_csv_loaded
    extracted_gid = client.extract_subgraph()
    start_vertices = 11
    max_depth = 2
    (vertex_paths, edge_weights, path_sizes) = client.node2vec(
        start_vertices, max_depth, extracted_gid
    )
    # FIXME: consider a more thorough test
    assert isinstance(vertex_paths, list) and len(vertex_paths)
    assert isinstance(edge_weights, list) and len(edge_weights)
    assert isinstance(path_sizes, list) and len(path_sizes)


def test_extract_subgraph(client_with_edgelist_csv_loaded):
    (client, test_data) = client_with_edgelist_csv_loaded
    Gid = client.extract_subgraph(
        create_using=None,
        selection=None,
        edge_weight_property="2",
        default_edge_weight=None,
        allow_multi_edges=False,
    )
    # FIXME: consider a more thorough test
    assert Gid in client.get_graph_ids()


def test_load_and_call_graph_creation_extension(
    client_with_graph_creation_extension_loaded, graph_creation_extension2
):
    """
    Tests calling a user-defined server-side graph creation extension from the
    cugraph_service client.
    """
    # The graph_creation_extension returns the tmp dir created which contains
    # the extension
    extension_dir = graph_creation_extension2
    client = client_with_graph_creation_extension_loaded

    num_files_loaded = client.load_graph_creation_extensions(extension_dir.name)
    assert num_files_loaded == 1

    new_graph_id = client.call_graph_creation_extension(
        "my_graph_creation_function", "a", "b", "c"
    )

    assert new_graph_id in client.get_graph_ids()

    # Inspect the PG and ensure it was created from my_graph_creation_function
    # FIXME: add client APIs to allow for a more thorough test of the graph
    assert client.get_graph_info(["num_edges"], new_graph_id) == 2

    # Ensure the other graph creation extension (loaded as part of
    # client_with_graph_creation_extension_loaded) can still be called
    new_graph_id = client.call_graph_creation_extension(
        "custom_graph_creation_function"
    )

    assert new_graph_id in client.get_graph_ids()


def test_load_and_call_graph_creation_long_running_extension(
    client_with_graph_creation_extension_loaded, graph_creation_extension_long_running
):
    """
    Tests calling a user-defined server-side graph creation extension from the
    cugraph_service client.
    """
    # The graph_creation_extension returns the tmp dir created which contains
    # the extension
    extension_dir = graph_creation_extension_long_running
    client = client_with_graph_creation_extension_loaded

    num_files_loaded = client.load_graph_creation_extensions(extension_dir.name)
    assert num_files_loaded == 1

    new_graph_id = client.call_graph_creation_extension(
        "long_running_graph_creation_function"
    )

    assert new_graph_id in client.get_graph_ids()

    # Inspect the PG and ensure it was created from my_graph_creation_function
    # FIXME: add client APIs to allow for a more thorough test of the graph
    assert client.get_graph_info(["num_edges"], new_graph_id) == 0


def test_call_graph_creation_extension(client_with_graph_creation_extension_loaded):
    """
    Ensure the graph creation extension preloaded by the server fixture is
    callable.
    """
    client = client_with_graph_creation_extension_loaded

    new_graph_id = client.call_graph_creation_extension(
        "custom_graph_creation_function"
    )

    assert new_graph_id in client.get_graph_ids()

    # Inspect the PG and ensure it was created from
    # custom_graph_creation_function
    # FIXME: add client APIs to allow for a more thorough test of the graph
    assert client.get_graph_info(["num_edges"], new_graph_id) == 3


def test_get_graph_vertex_data(client_with_property_csvs_loaded):
    (client, test_data) = client_with_property_csvs_loaded

    # FIXME: do not hardcode the shape values, get them from the input data.
    np_array_all_vertex_data = client.get_graph_vertex_data()
    assert np_array_all_vertex_data.shape == (9, 9)

    # The remaining tests get individual vertex data - compare those to the
    # all_vertex_data retrieved earlier.
    vert_ids = [11, 86, 89021]
    np_array = client.get_graph_vertex_data(vert_ids)
    assert np_array.shape == (3, 9)
    # The 1st element is the vert ID
    for (i, vid) in enumerate(vert_ids):
        assert np_array[i][0] == vid

    np_array = client.get_graph_vertex_data(11)
    assert np_array.shape == (1, 9)
    assert np_array[0][0] == 11

    np_array = client.get_graph_vertex_data(86)
    assert np_array.shape == (1, 9)
    assert np_array[0][0] == 86


def test_get_graph_edge_data(client_with_property_csvs_loaded):
    (client, test_data) = client_with_property_csvs_loaded

    # FIXME: do not hardcode the shape values, get them from the input data.
    np_array_all_rows = client.get_graph_edge_data()
    assert np_array_all_rows.shape == (17, 11)

    # The remaining tests get individual edge data - compare those to the
    # all_edge_data retrieved earlier.
    edge_ids = [0, 1, 2]
    np_array = client.get_graph_edge_data(edge_ids)
    assert np_array.shape == (3, 11)
    # The 0th element is the edge ID
    for (i, eid) in enumerate(edge_ids):
        assert np_array[i][0] == eid

    np_array = client.get_graph_edge_data(0)
    assert np_array.shape == (1, 11)
    assert np_array[0][0] == 0

    np_array = client.get_graph_edge_data(1)
    assert np_array.shape == (1, 11)
    assert np_array[0][0] == 1


def test_get_graph_info(client_with_property_csvs_loaded):
    (client, test_data) = client_with_property_csvs_loaded

    info = client.get_graph_info(["num_vertices", "num_vertex_properties"])
    data = (info["num_vertices"], info["num_vertex_properties"])
    # FIXME: do not hardcode values, get them from the input data.
    assert data == (9, 7)

    info = client.get_graph_info(["num_edges", "num_edge_properties"])
    data = (info["num_edges"], info["num_edge_properties"])
    # FIXME: do not hardcode values, get them from the input data.
    assert data == (17, 7)


def test_batched_ego_graphs(client_with_edgelist_csv_loaded):
    (client, test_data) = client_with_edgelist_csv_loaded

    extracted_gid = client.extract_subgraph()

    # These are known vertex IDs in the default graph loaded
    seeds = [0, 1, 2]
    results_lists = client.batched_ego_graphs(seeds, radius=1, graph_id=extracted_gid)

    (srcs, dsts, weights, seeds_offsets) = results_lists

    assert isinstance(srcs, Sequence)
    assert isinstance(dsts, Sequence)
    assert isinstance(weights, Sequence)
    assert len(srcs) == len(dsts) == len(weights)

    assert isinstance(seeds_offsets, Sequence)
    assert len(srcs) == seeds_offsets[-1]


def test_get_edge_IDs_for_vertices(client_with_edgelist_csv_loaded):
    (client, test_data) = client_with_edgelist_csv_loaded

    extracted_gid = client.extract_subgraph()

    srcs = [1, 2, 3]
    dsts = [0, 0, 0]

    edge_IDs = client.get_edge_IDs_for_vertices(srcs, dsts, graph_id=extracted_gid)

    assert len(edge_IDs) == len(srcs)


def test_uniform_neighbor_sampling(client_with_edgelist_csv_loaded):
    from cugraph_service_client.exceptions import CugraphServiceError
    from cugraph_service_client import defaults

    (client, test_data) = client_with_edgelist_csv_loaded

    start_list = [1, 2, 3]
    fanout_vals = [2, 2, 2]
    with_replacement = True

    # invalid graph type - default graph is a PG, needs an extracted subgraph
    with pytest.raises(CugraphServiceError):
        client.uniform_neighbor_sample(
            start_list=start_list,
            fanout_vals=fanout_vals,
            with_replacement=with_replacement,
            graph_id=defaults.graph_id,
        )

    extracted_gid = client.extract_subgraph(renumber_graph=True)
    # Ensure call can be made, assume results verified in other tests
    client.uniform_neighbor_sample(
        start_list=start_list,
        fanout_vals=fanout_vals,
        with_replacement=with_replacement,
        graph_id=extracted_gid,
    )
