# Copyright (c) 2021, NVIDIA CORPORATION.
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

import pytest

from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from dask_cuda.initialize import initialize

from cugraph.comms import comms as Comms
from cugraph.dask.common.mg_utils import get_visible_devices


# session-wide fixtures

@pytest.fixture(scope="session")
def dask_client():
    dask_scheduler_file = os.environ.get("SCHEDULER_FILE")
    cluster = None
    client = None
    if dask_scheduler_file:
        # Env var UCX_MAX_RNDV_RAILS=1 must be set too.
        initialize(enable_tcp_over_ucx=True,
                   enable_nvlink=True,
                   enable_infiniband=True,
                   enable_rdmacm=True,
                   #net_devices="mlx5_0:1",
                  )
        client = Client(scheduler_file=dask_scheduler_file)
        print(f"dask_client fixture: client created using {dask_scheduler_file}")
    else:
        # FIXME: use a better local_dir
        cluster = LocalCUDACluster(local_dir="/tmp")
        client = Client(cluster)
        client.wait_for_workers(len(get_visible_devices()))
        print(f"dask_client fixture: client created using LocalCUDACluster")

    Comms.initialize(p2p=True)

    yield client

    Comms.destroy()
    client.close()
    if cluster:
        cluster.close()
    print(f"dask_client fixture: client.close() called")
