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
import tempfile

import pytest

from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from dask_cuda.initialize import initialize

from cugraph.comms import comms as Comms
from cugraph.dask.common.mg_utils import get_visible_devices


# module-wide fixtures

@pytest.fixture(scope="module")
def dask_client():
    dask_scheduler_file = os.environ.get("SCHEDULER_FILE")
    cluster = None
    client = None
    tempdir_object = None

    if dask_scheduler_file:
        # Env var UCX_MAX_RNDV_RAILS=1 must be set too.
        initialize(enable_tcp_over_ucx=True,
                   enable_nvlink=True,
                   enable_infiniband=True,
                   enable_rdmacm=True,
                   # net_devices="mlx5_0:1",
                   )
        client = Client(scheduler_file=dask_scheduler_file)
        print("\ndask_client fixture: client created using "
              f"{dask_scheduler_file}")
    else:
        # The tempdir created by tempdir_object should be cleaned up once
        # tempdir_object goes out-of-scope and is deleted.
        tempdir_object = tempfile.TemporaryDirectory()
        cluster = LocalCUDACluster(local_directory=tempdir_object.name)
        client = Client(cluster)
        client.wait_for_workers(len(get_visible_devices()))
        print("\ndask_client fixture: client created using LocalCUDACluster")

    Comms.initialize(p2p=True)

    yield client

    Comms.destroy()
    # Shut down the connected scheduler and workers
    # therefore we will no longer rely on killing the dask cluster ID
    # for MNMG runs
    client.shutdown()
    if cluster:
        cluster.close()
    print("\ndask_client fixture: client.close() called")
