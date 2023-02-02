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

from dask.distributed import Client
from cugraph.dask.comms import comms as Comms
from cugraph.testing.mg_utils import stop_dask_client, start_dask_client


@pytest.fixture(scope="module")
def dask_client():
    dask_scheduler_file = os.environ.get("SCHEDULER_FILE")

    if dask_scheduler_file is not None:
        dask_client = Client(scheduler_file=dask_scheduler_file)
        dask_cluster = None
    else:
        dask_client, dask_cluster = start_dask_client(dask_worker_devices="0")

    if not Comms.is_initialized():
        Comms.initialize(p2p=True)

    yield dask_client

    stop_dask_client(dask_client, dask_cluster)
    print("\ndask_client fixture: client.close() called")
