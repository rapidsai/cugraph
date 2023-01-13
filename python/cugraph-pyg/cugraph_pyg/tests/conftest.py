# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

from dask_cuda.initialize import initialize as dask_initialize
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from cugraph.dask.comms import comms as Comms
from cugraph.dask.common.mg_utils import get_visible_devices
from cugraph.testing.mg_utils import stop_dask_client

import tempfile

# module-wide fixtures

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


@pytest.fixture(scope="module")
def dask_client():
    dask_scheduler_file = os.environ.get("SCHEDULER_FILE")
    cuda_visible_devices = get_visible_devices()

    if dask_scheduler_file is not None:
        dask_initialize()
        dask_client = Client(scheduler_file=dask_scheduler_file)
    else:
        # The tempdir created by tempdir_object should be cleaned up once
        # tempdir_object goes out-of-scope and is deleted.
        tempdir_object = tempfile.TemporaryDirectory()
        cluster = LocalCUDACluster(
            local_directory=tempdir_object.name,
            protocol="tcp",
            CUDA_VISIBLE_DEVICES=cuda_visible_devices,
        )

        dask_client = Client(cluster)
        dask_client.wait_for_workers(len(cuda_visible_devices))

    if not Comms.is_initialized():
        Comms.initialize(p2p=True)

    yield dask_client

    stop_dask_client(dask_client)
    print("\ndask_client fixture: client.close() called")
