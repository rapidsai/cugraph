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
from pathlib import Path

import pytest

from . import data


###############################################################################
# fixtures

@pytest.fixture(scope="module")
def mg_server():
    """
    Start a cugraph_service server that uses multiple GPUs via a dask
    configuration, then stop it when done with the fixture.
    """
    from cugraph_service_server import server
    from cugraph_service_client import CugraphServiceClient
    from cugraph_service_client.exceptions import CugraphServiceError

    server_file = server.__file__
    server_process = None
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

        dask_scheduler_file = os.environ.get("SCHEDULER_FILE")
        if dask_scheduler_file is None:
            raise EnvironmentError("Environment variable SCHEDULER_FILE must "
                                   "be set to the path to a dask scheduler "
                                   "json file")
        dask_scheduler_file = Path(dask_scheduler_file)
        if not dask_scheduler_file.exists():
            raise FileNotFoundError("env var SCHEDULER_FILE is set to "
                                    f"{dask_scheduler_file}, which does not "
                                    "exist.")

        # pytest will update sys.path based on the tests it discovers, and for
        # this source tree, an entry for the parent of this "tests" directory
        # will be added. The parent to this "tests" directory also allows
        # imports to find the cugraph_service sources, so in oder to ensure the
        # server that's started is also using the same sources, the PYTHONPATH
        # env should be set to the sys.path being used in this process.
        env_dict = os.environ.copy()
        env_dict["PYTHONPATH"] = ":".join(sys.path)

        with subprocess.Popen(
                [sys.executable, server_file,
                 "--host", host,
                 "--port", str(port),
                 "--dask-scheduler-file",
                 dask_scheduler_file],
                env=env_dict,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True) as server_process:
            try:
                print("\nLaunched cugraph_service server, waiting for it to "
                      "start...",
                      end="", flush=True)
                max_retries = 10
                retries = 0
                while retries < max_retries:
                    try:
                        client.uptime()
                        print("started.")
                        break
                    except CugraphServiceError:
                        time.sleep(1)
                        retries += 1
                if retries >= max_retries:
                    raise RuntimeError("error starting server")
            except Exception:
                print(server_process.stdout.read())
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
def client(mg_server):
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


###############################################################################
# tests

def test_get_default_graph_info(client_with_edgelist_csv_loaded):
    """
    Test to ensure various info on the default graph loaded from the specified
    fixture is correct.
    """
    (client, test_data) = client_with_edgelist_csv_loaded

    # get_graph_type() is a test/debug API which returns a string repr of the
    # graph type. Ideally, users should not need to know the graph type.
    assert "MG" in client._get_graph_type()

    assert client.get_graph_info(["num_edges"]) == test_data["num_edges"]
    assert client.get_server_info()["num_gpus"] > 1


def test_get_edge_IDs_for_vertices(client_with_edgelist_csv_loaded):
    """
    """
    (client, test_data) = client_with_edgelist_csv_loaded

    # get_graph_type() is a test/debug API which returns a string repr of the
    # graph type. Ideally, users should not need to know the graph type.
    assert "MG" in client._get_graph_type()

    graph_id = client.extract_subgraph(allow_multi_edges=True)
    client.get_edge_IDs_for_vertices([1, 2, 3], [0, 0, 0], graph_id)
