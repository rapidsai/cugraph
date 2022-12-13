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

from threading import Thread

import pytest

from cugraph_service_client import (
    CugraphServiceClient,
    defaults,
)
from cugraph_service_server.testing import utils

from cugraph_benchmarking import params

host = defaults.host
port = defaults.port
graph_scale = 26
edge_factor = 4
batch_size = 100
seed = 42
with_replacement = False

@pytest.fixture(scope="module")
def running_server_for_sampling_with_graph(request):
    (client, server_process) = (
        utils.ensure_running_server_for_sampling(host=host,
                                                 port=port,
                                                 dask_scheduler_file=None,
                                                 start_local_cuda_cluster=True)
    )
    num_edges = (2**graph_scale) * edge_factor
    gid = client.call_graph_creation_extension(
        "create_graph_from_rmat_generator",
        scale=graph_scale,
        num_edges=num_edges,
        seed=seed,
        mg=False,
    )

    yield gid

    client.delete_graph(gid)

    if server_process is not None:
        print("\nTerminating server...", end="", flush=True)
        server_process.terminate()
        server_process.wait(timeout=60)
        print("done.", flush=True)


def setup_sampling_client(host, port, graph_id):
    client = CugraphServiceClient(host, port)
    fanout_vals = [10, 25]
    start_list = client.call_extension(
        "gen_vertex_list", graph_id, batch_size, seed
    )

    def sampling_function(result_device):
        return client.uniform_neighbor_sample(
            start_list,
            fanout_vals,
            with_replacement,
            graph_id=graph_id,
            result_device=result_device,
        )

    return sampling_function

import time

################################################################################
# Benchmarks
@pytest.mark.parametrize("num_clients", params.num_clients.values())
def bench_cgs_client_scaling_individual_time(running_server_for_sampling_with_graph,
                                             num_clients):

    graph_id = running_server_for_sampling_with_graph

    # start n-1 clients running a sampling request in parallel, then run and
    # benchmark the last one separately.
    threads = []
    for _ in range(num_clients - 1):
        sampling_function = setup_sampling_client(host, port, graph_id)
        threads.append(Thread(target=sampling_function, args=[None]))

    sampling_function = setup_sampling_client(host, port, graph_id)
    [t.start() for t in threads]

    # FIXME: this benchmark is not using a benchmark fixture because the
    # benchmark fixture runs the function multiple times (even when warmup is
    # disabled), and by the time the last run is started, most of the other
    # clients have completed. This results in result that look as if multiple
    # clients do not affect the run time.
    #
    # Use the benchmark fixture once it can be run in a way where each run is
    # timed with other running clients.
    st = time.perf_counter_ns()
    results = sampling_function(None)
    print(f"TIME: {(time.perf_counter_ns() - st) / 1e9}s")
    assert len(results.sources) > 1000

    [t.join() for t in threads]


@pytest.mark.parametrize("num_clients", params.num_clients.values())
def bench_cgs_client_scaling_total_time(gpubenchmark,
                                        running_server_for_sampling_with_graph,
                                        num_clients):

    graph_id = running_server_for_sampling_with_graph

    # start n-1 clients running a sampling request in parallel, then run and
    # benchmark the last one separately.
    threads = []
    for _ in range(num_clients):
        sampling_function = setup_sampling_client(host, port, graph_id)
        threads.append(Thread(target=sampling_function, args=[None]))

    def func():
        [t.start() for t in threads]
        [t.join() for t in threads]

    gpubenchmark(func)
