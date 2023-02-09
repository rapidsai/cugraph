# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

from cugraph_service_server.testing import utils
from . import data


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
    host = "0.0.0.0"
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
        start_local_cuda_cluster = False
        if dask_scheduler_file is None:
            start_local_cuda_cluster = True
        else:
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
            start_local_cuda_cluster=start_local_cuda_cluster,
            dask_scheduler_file=dask_scheduler_file,
        )

        # yield control to the tests, cleanup on return
        yield

        print("\nTerminating server...", end="", flush=True)
        server_process.terminate()
        server_process.wait(timeout=60)
        print("done.", flush=True)


@pytest.fixture(scope="module")
def sg_server_on_device_1():
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
        server_process = utils.start_server_subprocess(
            host=host,
            port=port,
            env_additions={"CUDA_VISIBLE_DEVICES": "1"},
        )

        # yield control to the tests, cleanup on return
        yield

        print("\nTerminating server...", end="", flush=True)
        server_process.terminate()
        server_process.wait(timeout=60)
        print("done.", flush=True)


@pytest.fixture(scope="module")
def client_of_mg_server(mg_server):
    """
    Creates a client instance to the running server, closes the client when the
    fixture is no longer used by tests.
    """
    from cugraph_service_client import CugraphServiceClient, defaults

    client = CugraphServiceClient(defaults.host, defaults.port)

    # yield control to the tests, cleanup on return
    yield client

    client.close()


@pytest.fixture(scope="function")
def client_of_mg_server_with_edgelist_csv_loaded(client_of_mg_server):
    """
    Loads the karate CSV into the default graph on the server.
    """
    test_data = data.edgelist_csv_data["karate"]
    client = client_of_mg_server
    client.load_csv_as_edge_data(
        test_data["csv_file_name"],
        dtypes=test_data["dtypes"],
        vertex_col_names=["0", "1"],
        type_name="",
    )
    assert client.get_graph_ids() == [0]

    # yield control to the tests, cleanup on return
    yield (client, test_data)

    for gid in client.get_graph_ids():
        client.delete_graph(gid)


@pytest.fixture(scope="module")
def client_of_sg_server_on_device_1(sg_server_on_device_1):
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

    # yield control to the tests, cleanup on return
    yield client

    client.close()


@pytest.fixture(
    scope="module",
    params=[int(n) for n in [1e1, 1e3, 1e6, 1e9, 2e9]],
    ids=lambda p: f"bytes={p:.1e}",
)
def client_of_sg_server_on_device_1_with_test_array(
    request,
    client_of_sg_server_on_device_1,
):
    client = client_of_sg_server_on_device_1
    nbytes = request.param

    test_array_id = client._create_test_array(nbytes)

    # yield control to the tests, cleanup on return
    yield (client, test_array_id, nbytes)

    client._delete_test_array(test_array_id)


@pytest.fixture(scope="function")
def client_of_sg_server_on_device_1_large_property_graph_loaded(
    client_of_sg_server_on_device_1,
    graph_creation_extension_large_property_graph,
):
    client = client_of_sg_server_on_device_1
    server_extension_dir = graph_creation_extension_large_property_graph

    ext_mod_names = client.load_graph_creation_extensions(server_extension_dir)

    # Assume fixture that starts server on device 1 has the extension loaded
    # for creating large property graphs.
    new_graph_id = client.call_graph_creation_extension(
        "graph_creation_extension_large_property_graph"
    )

    assert new_graph_id in client.get_graph_ids()

    # yield control to the tests, cleanup on return
    yield (client, new_graph_id)

    client.delete_graph(new_graph_id)
    for mod_name in ext_mod_names:
        client.unload_extension_module(mod_name)


# This fixture is parametrized for different device IDs to test against, and
# simply returns the param value to the test using it.
@pytest.fixture(scope="module", params=[None, 0], ids=lambda p: f"device={p}")
def result_device_id(request):
    return request.param


###############################################################################
# tests


def test_get_default_graph_info(client_of_mg_server_with_edgelist_csv_loaded):
    """
    Test to ensure various info on the default graph loaded from the specified
    fixture is correct.
    """
    (client_of_mg_server, test_data) = client_of_mg_server_with_edgelist_csv_loaded

    # get_graph_type() is a test/debug API which returns a string repr of the
    # graph type. Ideally, users should not need to know the graph type.
    assert "MG" in client_of_mg_server._get_graph_type()

    assert client_of_mg_server.get_graph_info(["num_edges"]) == test_data["num_edges"]
    assert client_of_mg_server.get_server_info()["num_gpus"] > 1


def test_get_edge_IDs_for_vertices(client_of_mg_server_with_edgelist_csv_loaded):
    (client_of_mg_server, test_data) = client_of_mg_server_with_edgelist_csv_loaded

    # get_graph_type() is a test/debug API which returns a string repr of the
    # graph type. Ideally, users should not need to know the graph type.
    assert "MG" in client_of_mg_server._get_graph_type()

    graph_id = client_of_mg_server.extract_subgraph(check_multi_edges=True)
    client_of_mg_server.get_edge_IDs_for_vertices([1, 2, 3], [0, 0, 0], graph_id)


def test_device_transfer(
    benchmark,
    result_device_id,
    client_of_sg_server_on_device_1_with_test_array,
):
    (client, test_array_id, nbytes) = client_of_sg_server_on_device_1_with_test_array

    # device to host via RPC is too slow for large transfers, so skip
    if result_device_id is None and nbytes > 1e6:
        return

    bytes_returned = benchmark(
        client._receive_test_array,
        test_array_id,
        result_device=result_device_id,
    )

    # bytes_returned should be a cupy array of int8 values on
    # result_device_id, and each value should be 1.
    # Why not uint8 and value 255? Because when transferring data to a CPU
    # (result_device=None), Apache Thrift is used, which does not support
    # unsigned int types.
    assert len(bytes_returned) == nbytes
    if result_device_id is None:
        assert type(bytes_returned) is list
        assert False not in [n == 1 for n in bytes_returned]
    else:
        assert type(bytes_returned) is cp.ndarray
        assert (bytes_returned == cp.ones(nbytes, dtype="int8")).all()
        device_n = cp.cuda.Device(result_device_id)
        assert bytes_returned.device == device_n


def test_uniform_neighbor_sampling_result_on_device_error(
    client_of_sg_server_on_device_1_large_property_graph_loaded,
):
    """
    Ensure errors are handled properly when using device transfer
    """
    from cugraph_service_client.exceptions import CugraphServiceError

    (client, graph_id) = client_of_sg_server_on_device_1_large_property_graph_loaded
    extracted_graph_id = client.extract_subgraph(graph_id=graph_id)

    start_list = [0, 1, 2]
    fanout_vals = []  # should raise an exception
    with_replacement = False

    with pytest.raises(CugraphServiceError):
        client.uniform_neighbor_sample(
            start_list=start_list,
            fanout_vals=fanout_vals,
            with_replacement=with_replacement,
            graph_id=extracted_graph_id,
            result_device=0,
        )


def test_uniform_neighbor_sampling_result_on_device(
    benchmark,
    result_device_id,
    client_of_sg_server_on_device_1_large_property_graph_loaded,
):
    """
    Ensures uniform_neighbor_sample() results are transfered from the server to
    a specific client device when specified.
    """
    (client, graph_id) = client_of_sg_server_on_device_1_large_property_graph_loaded
    extracted_graph_id = client.extract_subgraph(graph_id=graph_id)

    start_list = [0, 1, 2]
    fanout_vals = [2]
    with_replacement = False

    result = benchmark(
        client.uniform_neighbor_sample,
        start_list=start_list,
        fanout_vals=fanout_vals,
        with_replacement=with_replacement,
        graph_id=extracted_graph_id,
        result_device=result_device_id,
    )

    assert len(result.sources) == len(result.destinations) == len(result.indices)
    dtype = type(result.sources)

    if result_device_id is None:
        # host memory
        assert dtype is list
    else:
        # device memory
        assert dtype is cp.ndarray
        device_n = cp.cuda.Device(result_device_id)
        assert result.sources.device == device_n


def test_call_extension_result_on_device_error(
    extension1, client_of_sg_server_on_device_1
):
    """
    Ensure errors are handled properly when using device transfer
    """
    from cugraph_service_client.exceptions import CugraphServiceError

    client = client_of_sg_server_on_device_1
    extension_dir = extension1
    array1_len = 1.23  # should raise an exception
    array2_len = 10

    ext_mod_names = client.load_extensions(extension_dir)

    with pytest.raises(CugraphServiceError):
        client.call_extension(
            "my_nines_function",
            array1_len,
            "int32",
            array2_len,
            "float64",
            result_device=0,
        )

    for mod_name in ext_mod_names:
        client.unload_extension_module(mod_name)


def test_call_extension_result_on_device(
    benchmark, extension1, result_device_id, client_of_sg_server_on_device_1
):
    client = client_of_sg_server_on_device_1
    extension_dir = extension1
    array1_len = int(1e5)
    array2_len = int(1e5)

    ext_mod_names = client.load_extensions(extension_dir)

    # my_nines_function in extension1 returns a list of two lists of 9's with
    # sizes and dtypes based on args.
    results = benchmark(
        client.call_extension,
        "my_nines_function",
        array1_len,
        "int32",
        array2_len,
        "float64",
        result_device=result_device_id,
    )

    if result_device_id is None:
        assert len(results) == 2
        assert len(results[0]) == array1_len
        assert len(results[1]) == array2_len
        assert type(results[0][0]) == int
        assert type(results[1][0]) == float
        assert results[0][0] == 9
        assert results[1][0] == 9.0
    else:
        # results will be a n-tuple where n is the number of arrays returned. The
        # n-tuple contains each array as a device array on result_device_id.
        assert isinstance(results, list)
        assert len(results) == 2

        device_n = cp.cuda.Device(result_device_id)
        assert isinstance(results[0], cp.ndarray)
        assert results[0].device == device_n
        assert results[0].tolist() == [9] * array1_len

        assert isinstance(results[1], cp.ndarray)
        assert results[1].device == device_n
        assert results[1].tolist() == [9.0] * array2_len

    for mod_name in ext_mod_names:
        client.unload_extension_module(mod_name)


def test_extension_adds_graph(
    extension_adds_graph, result_device_id, client_of_sg_server_on_device_1
):
    """
    Ensures an extension can create and add a graph to the server and return the
    new graph ID and other data.
    """
    extension_dir = extension_adds_graph
    client = client_of_sg_server_on_device_1

    ext_mod_names = client.load_extensions(extension_dir)

    # The extension will add a graph, compute a value based on the graph data,
    # and return the new graph ID and the result.
    graph_ids_before = client.get_graph_ids()

    val1 = 22
    val2 = 33.1
    results = client.call_extension(
        "my_extension", val1, val2, result_device=result_device_id
    )

    graph_ids_after = client.get_graph_ids()

    assert len(graph_ids_after) - len(graph_ids_before) == 1
    new_gid = (set(graph_ids_after) - set(graph_ids_before)).pop()
    assert len(results) == 2
    assert results[0] == new_gid
    expected_edge_ids = [0, 1, 2]
    expected_val = [n + val1 + val2 for n in expected_edge_ids]

    if result_device_id is None:
        assert results[1] == expected_val
    else:
        device_n = cp.cuda.Device(result_device_id)
        assert results[0].device == device_n
        assert results[1].device == device_n
        assert results[1].tolist() == expected_val

    # FIXME: much of this test could be in a fixture which ensures the extension
    # is unloaded from the server before returning
    for mod_name in ext_mod_names:
        client.unload_extension_module(mod_name)


def test_inside_asyncio_event_loop(
    client_of_sg_server_on_device_1_large_property_graph_loaded, result_device_id
):
    import asyncio

    client, graph_id = client_of_sg_server_on_device_1_large_property_graph_loaded

    start_list = [1, 2, 3]
    fanout_vals = [2, 2, 2]
    with_replacement = True

    async def uns():
        return client.uniform_neighbor_sample(
            start_list=start_list,
            fanout_vals=fanout_vals,
            with_replacement=with_replacement,
            graph_id=graph_id,
            result_device=result_device_id,
        )

    # ensure call succeeds; have confirmed this fails without fix in client
    assert asyncio.run(uns()) is not None
