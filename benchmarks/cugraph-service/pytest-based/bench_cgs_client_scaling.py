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

import pytest

from cugraph_service_client import (
    CugraphServiceClient,
    defaults,
)
from cugraph_service_server.testing import utils

from cugraph_benchmarking import params

host = defaults.host
port = defaults.port

@pytest.fixture(scope="module")
def running_server(request):
    (_, server_process) = utils.ensure_running_server(host=host,
                                                      port=port,
                                                      dask_scheduler_file=None,
                                                      start_local_cuda_cluster=True)
    yield
    if server_process is not None:
        print("\nTerminating server...", end="", flush=True)
        server_process.terminate()
        server_process.wait(timeout=60)
        print("done.", flush=True)


def start_sampling_thread():
    client = CugraphServiceClient(host, port)


################################################################################
# Benchmarks
@pytest.mark.parametrize("num_clients", params.num_clients.values())
def bench_cgs_client_scaling(running_server, num_clients):

    # start n-1 clients running a sampling request in parallel, then run and
    # benchmark the last one separately.
    threads = []
    for _ in range(num_clients - 1):
        threads.append(start_sampling_thread())

    nth_client = CugraphServiceClient()
