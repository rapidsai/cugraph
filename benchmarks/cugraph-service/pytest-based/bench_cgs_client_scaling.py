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

from threading import Thread

import pytest

from cugraph_service_client import (
    CugraphServiceClient,
    defaults,
)
from cugraph_service_server.testing import utils

from cugraph_benchmarking import params
from cugraph_benchmarking.timer import TimerContext


_host = defaults.host
_port = defaults.port
_graph_scale = 24
_edge_factor = 16
_batch_size = 100
_with_replacement = False
_rng_seed = 42


@pytest.fixture(scope="module")
def running_server_for_sampling_with_graph(request):
    """
    This fixture will yield a graph ID that can be used for the server at
    "_host" and "_port" defined in this module, which refers to a RMAT-generated
    graph of "_graph_scale" and "_edge_factor", also defined in this module.  If
    a cugraph-service server is running, it will use that, otherwise it will
    start a new server subprocess.  After the test/benchmark completes and
    control returns to this fixture, the graph is deleted. If the fixture
    started a server subprocess, then it is terminated as well.
    """
    (client, server_process) = (
        utils.ensure_running_server_for_sampling(host=_host,
                                                 port=_port,
                                                 dask_scheduler_file=None,
                                                 start_local_cuda_cluster=True)
    )
    num_edges = (2**_graph_scale) * _edge_factor
    gid = client.call_graph_creation_extension(
        "create_graph_from_rmat_generator",
        scale=_graph_scale,
        num_edges=num_edges,
        seed=_rng_seed,
        mg=False,
    )

    yield gid

    client.delete_graph(gid)

    if server_process is not None:
        print("\nTerminating server...", end="", flush=True)
        server_process.terminate()
        server_process.wait(timeout=60)
        print("done.", flush=True)


def create_sampling_client(host, port, graph_id):
    """
    Returns a callable that, when called will run uniform_neighbor_sample. This
    is intended to be called in a separate thread to produce multiple
    concurrent client sampling calls.
    """
    client = CugraphServiceClient(host, port)
    fanout_vals = [10, 25]
    start_list = client.call_extension(
        "gen_vertex_list", graph_id, _batch_size
    )

    def sampling_function(result_device):
        return client.uniform_neighbor_sample(
            start_list,
            fanout_vals,
            _with_replacement,
            graph_id=graph_id,
            result_device=result_device,
        )

    return sampling_function


################################################################################
# Benchmarks

# FIXME: these benchmarks are not using a benchmark fixture because the
# benchmark fixture runs the function multiple times (even when warmup is
# disabled), and by the time the last run is started, most of the other clients
# have completed. This results in result that look as if multiple clients do
# not affect the run time.

# Use the benchmark fixture once it can be run in a way where each run is timed
# with other running clients.

@pytest.mark.parametrize("num_clients", params.num_clients.values())
def bench_cgs_client_scaling_individual_time(running_server_for_sampling_with_graph,
                                             num_clients):

    graph_id = running_server_for_sampling_with_graph

    # start n-1 clients running a sampling request in parallel, then run and
    # benchmark the last one separately.
    threads = []
    for _ in range(num_clients - 1):
        sampling_function = create_sampling_client(_host, _port, graph_id)
        threads.append(Thread(target=sampling_function, args=[None]))

    sampling_function = create_sampling_client(_host, _port, graph_id)
    [t.start() for t in threads]

    with TimerContext():
        results = sampling_function(None)

    assert len(results.sources) > 1000

    [t.join() for t in threads]


@pytest.mark.parametrize("num_clients", params.num_clients.values())
def bench_cgs_client_scaling_total_time(running_server_for_sampling_with_graph,
                                        num_clients):

    graph_id = running_server_for_sampling_with_graph

    threads = []
    for _ in range(num_clients):
        sampling_function = create_sampling_client(_host, _port, graph_id)
        threads.append(Thread(target=sampling_function, args=[None]))

    with TimerContext():
        [t.start() for t in threads]
        [t.join() for t in threads]
