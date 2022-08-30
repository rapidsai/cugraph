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

import os
import sys
import subprocess
import time
from collections.abc import Sequence

import pytest

from . import data


###############################################################################
## fixtures

@pytest.fixture(scope="module")
def server(graph_creation_extension1):
    """
    Start a GaaS server, stop it when done with the fixture.  This also uses
    graph_creation_extension1 to preload a graph creation extension.
    """
    from gaas_server import server
    from gaas_client import GaasClient
    from gaas_client.exceptions import GaasError

    server_file = server.__file__
    server_process = None
    host = "localhost"
    port = 9090
    graph_creation_extension_dir = graph_creation_extension1
    client = GaasClient(host, port)

    try:
        client.uptime()
        print("\nfound running server, assuming it should be used for testing!")
        yield

    except GaasError:
        # A server was not found, so start one for testing then stop it when
        # testing is done.

        # pytest will update sys.path based on the tests it discovers, and for
        # this source tree, an entry for the parent of this "tests" directory
        # will be added. The parent to this "tests" directory also allows
        # imports to find the GaaS sources, so in oder to ensure the server
        # that's started is also using the same sources, the PYTHONPATH env
        # should be set to the sys.path being used in this process.
        env_dict = os.environ.copy()
        env_dict["PYTHONPATH"] = ":".join(sys.path)

        with subprocess.Popen(
                [sys.executable, server_file,
                 "--host", host,
                 "--port", str(port),
                 "--graph-creation-extension-dir",
                 graph_creation_extension_dir],
                env=env_dict,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True) as server_process:
            try:
                print("\nLaunched GaaS server, waiting for it to start...",
                      end="", flush=True)
                max_retries = 10
                retries = 0
                while retries < max_retries:
                    try:
                        client.uptime()
                        print("started.")
                        break
                    except GaasError:
                        time.sleep(1)
                        retries += 1
                if retries >= max_retries:
                    raise RuntimeError("error starting server")
            except:
                if server_process.poll() is None:
                    server_process.terminate()
                raise

            # yield control to the tests
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
    from gaas_client import GaasClient, defaults

    client = GaasClient(defaults.host, defaults.port)

    for gid in client.get_graph_ids():
        client.delete_graph(gid)

    #client.unload_graph_creation_extensions()

    # yield control to the tests
    yield client

    # tests are done, now stop the server
    client.close()


@pytest.fixture(scope="function")
def client_with_edgelist_csv_loaded(client):
    """
    Loads the karate CSV into the default graph on the server.
    """
    test_data = data.edgelist_csv_data["karate"]
    client.load_csv_as_edge_data(test_data["csv_file_name"],
                                 dtypes=test_data["dtypes"],
                                 vertex_col_names=["0", "1"],
                                 type_name="")
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

    client.load_csv_as_vertex_data(merchants["csv_file_name"],
                                   dtypes=merchants["dtypes"],
                                   vertex_col_name=merchants["vert_col_name"],
                                   header=0,
                                   type_name="merchants")
    client.load_csv_as_vertex_data(users["csv_file_name"],
                                   dtypes=users["dtypes"],
                                   vertex_col_name=users["vert_col_name"],
                                   header=0,
                                   type_name="users")

    client.load_csv_as_edge_data(transactions["csv_file_name"],
                                 dtypes=transactions["dtypes"],
                                 vertex_col_names=\
                                 transactions["vert_col_names"],
                                 header=0,
                                 type_name="transactions")
    client.load_csv_as_edge_data(relationships["csv_file_name"],
                                 dtypes=relationships["dtypes"],
                                 vertex_col_names=\
                                 relationships["vert_col_names"],
                                 header=0,
                                 type_name="relationships")
    client.load_csv_as_edge_data(referrals["csv_file_name"],
                                 dtypes=referrals["dtypes"],
                                 vertex_col_names=referrals["vert_col_names"],
                                 header=0,
                                 type_name="referrals")

    assert client.get_graph_ids() == [0]
    return (client, data.property_csv_data)


###############################################################################
## tests
def test_get_graph_info_key_types(client_with_property_csvs_loaded):
    """
    Tests error handling for info keys passed in.
    """
    from gaas_client.exceptions import GaasError

    (client, test_data) = client_with_property_csvs_loaded

    with pytest.raises(TypeError):
        client.get_graph_info(21)  # bad key type
    with pytest.raises(TypeError):
        client.get_graph_info([21, "num_edges"])  # bad key type
    with pytest.raises(GaasError):
        client.get_graph_info("21")  # bad key value
    with pytest.raises(GaasError):
        client.get_graph_info(["21"])  # bad key value
    with pytest.raises(GaasError):
        client.get_graph_info(["num_edges", "21"])  # bad key value

    client.get_graph_info()  # valid

def test_get_num_edges_default_graph(client_with_edgelist_csv_loaded):
    (client, test_data) = client_with_edgelist_csv_loaded
    assert client.get_graph_info("num_edges") == test_data["num_edges"]

def test_load_csv_as_edge_data_nondefault_graph(client):
    from gaas_client.exceptions import GaasError

    test_data = data.edgelist_csv_data["karate"]

    with pytest.raises(GaasError):
        client.load_csv_as_edge_data(test_data["csv_file_name"],
                                     dtypes=test_data["dtypes"],
                                     vertex_col_names=["0", "1"],
                                     type_name="",
                                     graph_id=9999)

def test_get_num_edges_nondefault_graph(client_with_edgelist_csv_loaded):
    from gaas_client.exceptions import GaasError

    (client, test_data) = client_with_edgelist_csv_loaded
    # Bad graph ID
    with pytest.raises(GaasError):
        client.get_graph_info("num_edges", graph_id=9999)

    new_graph_id = client.create_graph()
    client.load_csv_as_edge_data(test_data["csv_file_name"],
                                 dtypes=test_data["dtypes"],
                                 vertex_col_names=["0", "1"],
                                 type_name="",
                                 graph_id=new_graph_id)

    assert client.get_graph_info("num_edges") == test_data["num_edges"]
    assert client.get_graph_info("num_edges", graph_id=new_graph_id) \
        == test_data["num_edges"]


def test_node2vec(client_with_edgelist_csv_loaded):
    (client, test_data) = client_with_edgelist_csv_loaded
    extracted_gid = client.extract_subgraph()
    start_vertices = 11
    max_depth = 2
    (vertex_paths, edge_weights, path_sizes) = \
        client.node2vec(start_vertices, max_depth, extracted_gid)
    # FIXME: consider a more thorough test
    assert isinstance(vertex_paths, list) and len(vertex_paths)
    assert isinstance(edge_weights, list) and len(edge_weights)
    assert isinstance(path_sizes, list) and len(path_sizes)


def test_extract_subgraph(client_with_edgelist_csv_loaded):
    (client, test_data) = client_with_edgelist_csv_loaded
    Gid = client.extract_subgraph(create_using=None,
                                  selection=None,
                                  edge_weight_property="2",
                                  default_edge_weight=None,
                                  allow_multi_edges=False)
    # FIXME: consider a more thorough test
    assert Gid in client.get_graph_ids()


def test_load_and_call_graph_creation_extension(client,
                                                graph_creation_extension2):
    """
    Tests calling a user-defined server-side graph creation extension from the
    GaaS client.
    """
    # The graph_creation_extension returns the tmp dir created which contains
    # the extension
    extension_dir = graph_creation_extension2

    num_files_loaded = client.load_graph_creation_extensions(extension_dir)
    assert num_files_loaded == 1

    new_graph_ID = client.call_graph_creation_extension(
        "my_graph_creation_function", "a", "b", "c")

    assert new_graph_ID in client.get_graph_ids()

    # Inspect the PG and ensure it was created from my_graph_creation_function
    # FIXME: add client APIs to allow for a more thorough test of the graph
    assert client.get_graph_info(["num_edges"], new_graph_ID) == 2


def test_load_and_call_graph_creation_long_running_extension(
        client,
        graph_creation_extension_long_running):
    """
    Tests calling a user-defined server-side graph creation extension from the
    GaaS client.
    """
    # The graph_creation_extension returns the tmp dir created which contains
    # the extension
    extension_dir = graph_creation_extension_long_running

    num_files_loaded = client.load_graph_creation_extensions(extension_dir)
    assert num_files_loaded == 1

    new_graph_ID = client.call_graph_creation_extension(
        "long_running_graph_creation_function")

    assert new_graph_ID in client.get_graph_ids()

    # Inspect the PG and ensure it was created from my_graph_creation_function
    # FIXME: add client APIs to allow for a more thorough test of the graph
    assert client.get_graph_info(["num_edges"], new_graph_ID) == 0


def test_call_graph_creation_extension(client):
    """
    Ensure the graph creation extension preloaded by the server fixture is
    callable.
    """
    new_graph_ID = client.call_graph_creation_extension(
        "custom_graph_creation_function")

    assert new_graph_ID in client.get_graph_ids()

    # Inspect the PG and ensure it was created from
    # custom_graph_creation_function
    # FIXME: add client APIs to allow for a more thorough test of the graph
    assert client.get_graph_info(["num_edges"], new_graph_ID) == 3


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
    # The 3rd element is the edge ID
    for (i, eid) in enumerate(edge_ids):
        assert np_array[i][2] == eid

    np_array = client.get_graph_edge_data(0)
    assert np_array.shape == (1, 11)
    assert np_array[0][2] == 0

    np_array = client.get_graph_edge_data(1)
    assert np_array.shape == (1, 11)
    assert np_array[0][2] == 1


def test_get_graph_info(client_with_property_csvs_loaded):
    (client, test_data) = client_with_property_csvs_loaded

    info = client.get_graph_info(["num_vertices",
                                  "num_vertex_properties"])
    data = (info["num_vertices"],
            info["num_vertex_properties"])
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
    results_lists = client.batched_ego_graphs(
        seeds, radius=1, graph_id=extracted_gid)

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

    edge_IDs = client.get_edge_IDs_for_vertices(srcs, dsts,
                                                graph_id=extracted_gid)

    assert len(edge_IDs) == len(srcs)


def test_uniform_neighbor_sampling(client_with_edgelist_csv_loaded):
    from gaas_client.exceptions import GaasError
    from gaas_client import defaults

    (client, test_data) = client_with_edgelist_csv_loaded

    start_list = [1, 2, 3]
    fanout_vals = [2, 2, 2]
    with_replacement = True

    # invalid graph type - default graph is a PG, needs an extracted subgraph
    with pytest.raises(GaasError):
        client.uniform_neighbor_sample(start_list=start_list,
                                       fanout_vals=fanout_vals,
                                       with_replacement=with_replacement,
                                       graph_id=defaults.graph_id)

    extracted_gid = client.extract_subgraph(renumber_graph=True)
    result = client.uniform_neighbor_sample(start_list=start_list,
                                            fanout_vals=fanout_vals,
                                            with_replacement=with_replacement,
                                            graph_id=extracted_gid)
