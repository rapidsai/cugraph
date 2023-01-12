# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from dask_cuda.initialize import initialize

from cugraph.dask.comms import comms as Comms
from cugraph.dask.common.mg_utils import get_visible_devices


def start_dask_client(
    protocol=None,
    rmm_pool_size=None,
    dask_worker_devices=None,
):
    dask_scheduler_file = os.environ.get("SCHEDULER_FILE")
    cluster = None
    client = None
    tempdir_object = None

    if dask_scheduler_file:
        if protocol is not None:
            print(
                f"WARNING: {protocol=} is ignored in start_dask_client() when using "
                "dask SCHEDULER_FILE"
            )
        if rmm_pool_size is not None:
            print(
                f"WARNING: {rmm_pool_size=} is ignored in start_dask_client() when "
                "using dask SCHEDULER_FILE"
            )
        if dask_worker_devices is not None:
            print(
                f"WARNING: {dask_worker_devices=} is ignored in start_dask_client() "
                "when using dask SCHEDULER_FILE"
            )

        initialize()
        client = Client(scheduler_file=dask_scheduler_file)
        print("\ndask_client created using " f"{dask_scheduler_file}")
    else:
        # The tempdir created by tempdir_object should be cleaned up once
        # tempdir_object goes out-of-scope and is deleted.
        tempdir_object = tempfile.TemporaryDirectory()
        cluster = LocalCUDACluster(
            local_directory=tempdir_object.name,
            protocol=protocol,
            rmm_pool_size=rmm_pool_size,
            CUDA_VISIBLE_DEVICES=dask_worker_devices,
        )
        client = Client(cluster)
        client.wait_for_workers(len(get_visible_devices()))
        print("\ndask_client created using LocalCUDACluster")

    Comms.initialize(p2p=True)

    return (client, cluster)


def stop_dask_client(client, cluster=None):
    Comms.destroy()
    client.close()
    if cluster:
        cluster.close()
    print("\ndask_client closed.")
