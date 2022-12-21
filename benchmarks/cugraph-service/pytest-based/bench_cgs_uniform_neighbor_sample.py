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

import time
import os
from pathlib import Path

import pytest
import numpy as np

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

from cugraph_service_client import CugraphServiceClient
from cugraph_service_client.exceptions import CugraphServiceError
from cugraph_service_client import RemoteGraph
from cugraph_service_server.testing import utils

from cugraph_benchmarking import params

_seed = 42


def create_remote_graph(graph_data, is_mg, client):
    """
    Create a remote graph instance based on the data to be loaded/generated,
    relying on server-side graph creation extensions.

    The server extension is part of the
    "cugraph_service_server.testing.benchmark_server_extension" and is loaded
    in the ensure_running_service_for_sampling() helper.
    """
    # Assume strings are names of datasets in the datasets package
    if isinstance(graph_data, str):
        gid = client.call_graph_creation_extension(
            "create_graph_from_builtin_dataset", graph_data
        )

    # Assume dictionary contains RMAT params
    elif isinstance(graph_data, dict):
        scale = graph_data["scale"]
        num_edges = (2**scale) * graph_data["edgefactor"]
        seed = _seed
        gid = client.call_graph_creation_extension(
            "create_graph_from_rmat_generator",
            scale=scale,
            num_edges=num_edges,
            seed=seed,
            mg=is_mg,
        )
    else:
        raise TypeError(f"graph_data can only be str or dict, got {type(graph_data)}")

    G = RemoteGraph(client, gid)
    return G


def get_uniform_neighbor_sample_args(
    G, seed, batch_size, fanout, with_replacement
):
    """
    Return a dictionary containing the args for uniform_neighbor_sample based
    on the graph and desired args passed in. For example, if a large start list
    and small fanout list is desired, a "large" (based on graph size) list of
    valid vert IDs for the graph passed in and a "small" list of fanout values
    will be returned.

    The dictionary return value allows for easily supporting other args without
    having to maintain an order of values in a return tuple, for example.
    """
    if with_replacement not in [True, False]:
        raise ValueError(f"got unexpected value {with_replacement=}")

    num_verts = G.number_of_vertices()

    if batch_size > num_verts:
        num_start_verts = int(num_verts * 0.25)
    else:
        num_start_verts = batch_size

    # Create the start_list on the server, since generating a list of actual
    # IDs requires unrenumbering steps that cannot be easily done remotely.
    start_list = G._client.call_extension(
        "gen_vertex_list", G._graph_id, num_start_verts, seed
    )

    return {
        "start_list": list(start_list),
        "fanout": fanout,
        "with_replacement": with_replacement,
    }


def ensure_running_service_for_sampling(dask_scheduler_file=None,
                                        start_local_cuda_cluster=False):
    """
    Returns a tuple containing a Popen object for the running cugraph-service
    server subprocess, and a client object connected to it.  If a server was
    detected already running, the Popen object will be None.
    """
    host = "localhost"
    port = 9090
    client = CugraphServiceClient(host, port)
    server_process = None

    try:
        client.uptime()
        print("FOUND RUNNING SERVER, ASSUMING IT SHOULD BE USED FOR TESTING!")

    except CugraphServiceError:
        # A server was not found, so start one for testing then stop it when
        # testing is done.
        server_process = utils.start_server_subprocess(
            host=host,
            port=port,
            start_local_cuda_cluster=start_local_cuda_cluster,
            dask_scheduler_file=dask_scheduler_file,
        )

    # Ensure the extensions needed for these benchmarks are loaded
    required_graph_creation_extension_module = "benchmark_server_extension"
    server_data = client.get_server_info()
    # .stem excludes .py extensions, so it can match a python module name
    loaded_graph_creation_extension_modules = [
        Path(m).stem for m in server_data["graph_creation_extensions"]
    ]
    if (
        required_graph_creation_extension_module
        not in loaded_graph_creation_extension_modules
    ):
        modules_loaded = client.load_graph_creation_extensions(
            "cugraph_service_server.testing.benchmark_server_extension"
        )
        if len(modules_loaded) < 1:
            raise RuntimeError(
                "failed to load graph creation extension "
                f"{required_graph_creation_extension_module}"
            )

    loaded_extension_modules = [Path(m).stem for m in server_data["extensions"]]
    if required_graph_creation_extension_module not in loaded_extension_modules:
        modules_loaded = client.load_extensions(
            "cugraph_service_server.testing.benchmark_server_extension"
        )
        if len(modules_loaded) < 1:
            raise RuntimeError(
                "failed to load extension "
                f"{required_graph_creation_extension_module}"
            )

    return (server_process, client)


def remote_uniform_neighbor_sample(G, start_list, fanout_vals, with_replacement=True):
    """
    Calls uniform_neighbor_sample() on the server using the client assigned to
    the RemoteGraph instance G.
    """
    assert G.is_remote()
    result = G._client.uniform_neighbor_sample(
        start_list, fanout_vals, with_replacement, graph_id=G._graph_id, result_device=1
    )
    return result


@pytest.fixture(scope="module", params=params.graph_obj_fixture_params)
def remote_graph_objs(request):
    """
    Fixture that returns a RemoteGraph object populated with graph data and
    algo callable based on the parameters. This also ensures a cugraph-service
    server if not.
    """
    (gpu_config, graph_data) = request.param
    server_process = None

    if gpu_config not in ["SG", "SNMG", "MNMG"]:
        raise RuntimeError(f"got unexpected gpu_config value: {gpu_config}")

    # Ensure the appropriate server is running
    if gpu_config == "SG":
        (server_process, cgs_client) = ensure_running_service_for_sampling()
        is_mg = False

    elif gpu_config == "SNMG":
        dask_scheduler_file = os.environ.get("SCHEDULER_FILE")
        if dask_scheduler_file is None:
            (server_process, cgs_client) = ensure_running_service_for_sampling(
                start_local_cuda_cluster=True
            )
        else:
            assert Path(dask_scheduler_file).exists()
            (server_process, cgs_client) = ensure_running_service_for_sampling(
                dask_scheduler_file=dask_scheduler_file
            )
        is_mg = True

    else:
        raise NotImplementedError(f"{gpu_config=}")

    print("creating graph...")
    st = time.perf_counter_ns()
    G = create_remote_graph(graph_data, is_mg, cgs_client)
    print(f"done creating graph, took {((time.perf_counter_ns() - st) / 1e9)}s")

    uns_func = remote_uniform_neighbor_sample

    yield (G, uns_func)

    del G  # is this necessary?
    if server_process is not None:
        print("\nTerminating server...", end="", flush=True)
        server_process.terminate()
        server_process.wait(timeout=60)
        print("done.", flush=True)


################################################################################
# Benchmarks
@pytest.mark.parametrize("batch_size", params.batch_sizes.values())
@pytest.mark.parametrize("fanout", [params.fanout_10_25, params.fanout_5_10_15])
@pytest.mark.parametrize(
    "with_replacement", [False], ids=lambda v: f"with_replacement={v}"
)
def bench_cgs_uniform_neighbor_sample(
    gpubenchmark, remote_graph_objs, batch_size, fanout, with_replacement
):
    (G, uniform_neighbor_sample_func) = remote_graph_objs

    uns_args = get_uniform_neighbor_sample_args(
        G, _seed, batch_size, fanout, with_replacement
    )
    # print(f"\n{uns_args}")
    # FIXME: uniform_neighbor_sample cannot take a np.ndarray for start_list
    result = gpubenchmark(
        uniform_neighbor_sample_func,
        G,
        start_list=uns_args["start_list"],
        fanout_vals=uns_args["fanout"],
        with_replacement=uns_args["with_replacement"],
    )
    dtmap = {"int32": 32 // 8, "int64": 64 // 8}
    dt = str(result.sources.dtype)
    llen = len(result.sources)
    print(f"\nresult list len: {llen} (x3), dtype={dt}, total bytes={3*llen*dtmap[dt]}")
