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
from pathlib import Path

import pytest
import cupy as cp

from . import data
from . import utils


###############################################################################
# fixtures


@pytest.fixture(scope="module")
def mg_server():
    """
    Start a cugraph_service server that uses multiple GPUs via a dask
    configuration, then stop it when done with the fixture.

    This requires that a dask scheduler be running, and the corresponding
    SCHEDULER_FILE env var is set.  The scheduler can be started using the
    script in this repo:
    "<repo>/python/cugraph_service/scripts/run-dask-process.sh scheduler
    workers"
    """
    from cugraph_service_client import CugraphServiceClient
    from cugraph_service_client.exceptions import CugraphServiceError

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
            raise EnvironmentError(
                "Environment variable SCHEDULER_FILE must "
                "be set to the path to a dask scheduler "
                "json file"
            )
        dask_scheduler_file = Path(dask_scheduler_file)
        if not dask_scheduler_file.exists():
            raise FileNotFoundError(
                "env var SCHEDULER_FILE is set to "
                f"{dask_scheduler_file}, which does not "
                "exist."
            )

        server_process = utils.start_server_subprocess(
            host=host,
            port=port,
            dask_scheduler_file=dask_scheduler_file)

        # yield control to the tests
        yield

        # tests are done, now stop the server
        print("\nTerminating server...", end="", flush=True)
        server_process.terminate()
        print("done.", flush=True)


@pytest.fixture(scope="module")
def sg_server_on_device_1(graph_creation_extension_large_property_graph):
    """
    Start a cugraph_service server, stop it when done with the fixture.  This
    also uses graph_creation_extension_large_property_graph to preload the
    graph creation extension that creates a large PG.
    """
    from cugraph_service_client import CugraphServiceClient
    from cugraph_service_client.exceptions import CugraphServiceError

    host = "localhost"
    port = 9090
    server_extension_dir = graph_creation_extension_large_property_graph
    client = CugraphServiceClient(host, port)

    try:
        client.uptime()
        print("FOUND RUNNING SERVER, ASSUMING IT SHOULD BE USED FOR TESTING!")
        yield

    except CugraphServiceError:
        # A server was not found, so start one for testing then stop it when
        # testing is done.
        server_process = utils.start_server_subprocess(
            host=host,
            port=port,
            graph_creation_extension_dir=server_extension_dir.name,
            env_additions={"CUDA_VISIBLE_DEVICES": "1"}
        )

        # yield control to the tests
        yield

        # tests are done, now stop the server
        print("\nTerminating server...", end="", flush=True)
        server_process.terminate()
        print("done.", flush=True)


@pytest.fixture(scope="module")
def client(mg_server):
    """
    Creates a client instance to the running server, closes the client when the
    fixture is no longer used by tests.
    """
    from cugraph_service_client import CugraphServiceClient, defaults

    client = CugraphServiceClient(defaults.host, defaults.port)

    # yield control to the tests
    yield client

    # tests are done, now close the connection
    client.close()


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

    yield (client, test_data)

    # Cleanup after the test
    for gid in client.get_graph_ids():
        client.delete_graph(gid)


@pytest.fixture(scope="module")
def client_of_server_on_device_1(sg_server_on_device_1):
    """
    Creates a client instance to a server running on device 1, closes the
    client when the fixture is no longer used by tests.
    """
    from cugraph_service_client import CugraphServiceClient, defaults

    client = CugraphServiceClient(defaults.host, defaults.port)

    for gid in client.get_graph_ids():
        client.delete_graph(gid)

    # FIXME: should this fixture always unconditionally unload all extensions?
    # client.unload_graph_creation_extensions()

    # yield control to the tests
    yield client

    # tests are done, now close the connection
    client.close()

import time
@pytest.fixture(scope="module",
                params=[int(n) for n in [1e1, 1e3, 1e6, 1e9, 2e9, 5e9, 10e9]],
                ids=lambda p: f"bytes={p:.1e}")
def client_of_server_on_device_1_with_test_array(
        request,
        sg_server_on_device_1,
):
    from cugraph_service_client import CugraphServiceClient, defaults

    client = CugraphServiceClient(defaults.host, defaults.port)
    nbytes = request.param
    print(f"\nCREATING TEST ARRAY nbytes={nbytes:.1e}",flush=True)
    st=time.time()
    test_array_id = client._create_test_array(nbytes)
    print(f"DONE CREATING TEST ARRAY, time: {time.time()-st} s",flush=True)

    yield (client, test_array_id, nbytes)

    client._delete_test_array(test_array_id)
    client.close()

@pytest.fixture(scope="function")
def client_of_server_on_device_1_large_property_graph_loaded(
        client_of_server_on_device_1
):
    client = client_of_server_on_device_1
    # Assume fixture that starts server on device 1 has the extension loaded
    # for creating large property graphs.
    new_graph_id = client.call_graph_creation_extension(
        "graph_creation_extension_large_property_graph")

    assert new_graph_id in client.get_graph_ids()
    # yield control to the tests that use this fixture
    yield (client, new_graph_id)
    # all tests using this fixture are done, so delete the large graph
    client.delete_graph(new_graph_id)


# Because pytest does not allow mixing fixtures and parametrization decorators
# for test functions, this fixture is parametrized for different device IDs to
# test against, and simply returns the param value to the test using it.
@pytest.fixture(scope="module",
                params=[None, 0],
                ids=lambda p: f"device={p}")
def result_device_id(request):
    return request.param


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
    """ """
    (client, test_data) = client_with_edgelist_csv_loaded

    # get_graph_type() is a test/debug API which returns a string repr of the
    # graph type. Ideally, users should not need to know the graph type.
    assert "MG" in client._get_graph_type()

    graph_id = client.extract_subgraph(allow_multi_edges=True)
    client.get_edge_IDs_for_vertices([1, 2, 3], [0, 0, 0], graph_id)


def test_device_transfer(
        benchmark,
        result_device_id,
        client_of_server_on_device_1_with_test_array,
):
    (client, test_array_id, nbytes) = (
        client_of_server_on_device_1_with_test_array
    )

    # device to host via RPC is too slow for large transfers, so skip
    if result_device_id is None and nbytes > 1e6:
        return

    print("RUNNING BENCHMARK...",flush=True)
    st=time.time()
    bytes_returned = benchmark(client._receive_test_array,
                               test_array_id,
                               result_device=result_device_id,
                               )
    print(f"DONE RUNNING BENCHMARK, time: {time.time()-st}",flush=True)

    # bytes_returned should be a cupy array of int8 values on
    # result_device_id, and each value should be 1.
    # Why not uint8 and value 255? Because when transferring data to a CPU
    # (result_device=None), Apache Thrift is used, which does not support
    # unsigned int types.
    print("CHECKING RESULTS...",flush=True)
    st=time.time()
    assert len(bytes_returned) == nbytes
    if result_device_id is None:
        assert type(bytes_returned) is list
        assert False not in [n == 1 for n in bytes_returned]
    else:
        assert type(bytes_returned) is cp.ndarray
        assert (bytes_returned == cp.ones(nbytes, dtype="int8")).all()
        device_n = cp.cuda.Device(result_device_id)
        assert bytes_returned.device == device_n
    print(f"DONE CHECKING RESULTS, time: {time.time()-st}",flush=True)

def test_uniform_neighbor_sampling_result_device(
        benchmark,
        result_device_id,
        client_of_server_on_device_1_large_property_graph_loaded,
):
    """
    Ensures uniform_neighbor_sample() results are transfered from the server to
    a specific client device when specified.
    """
    (client, graph_id) = (
        client_of_server_on_device_1_large_property_graph_loaded
    )
    extracted_graph_id = client.extract_subgraph(graph_id=graph_id)

    start_list = range(int(1e3))
    fanout_vals = [2]
    with_replacement = False

    result = benchmark(
        client.uniform_neighbor_sample,
        start_list=start_list,
        fanout_vals=fanout_vals,
        with_replacement=with_replacement,
        graph_id=extracted_graph_id,
        result_device=result_device_id)

    assert (len(result.sources) ==
            len(result.destinations) ==
            len(result.indices)
            )
    dtype = type(result.sources)

    if result_device_id is None:
        # host memory
        assert dtype is list
    else:
        # device memory
        assert dtype is cp.ndarray
        device_n = cp.cuda.Device(result_device_id)
        assert result.sources.device == device_n
