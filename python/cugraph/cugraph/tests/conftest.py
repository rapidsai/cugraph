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

import pytest
from cugraph.testing.mg_utils import (
    start_dask_client,
    stop_dask_client,
)

import os
import tempfile

# module-wide fixtures


# Spoof the gpubenchmark fixture if it's not available so that asvdb and
# rapids-pytest-benchmark do not need to be installed to run tests.
if "gpubenchmark" not in globals():

    def benchmark_func(func, *args, **kwargs):
        return func(*args, **kwargs)

    @pytest.fixture
    def gpubenchmark():
        return benchmark_func


@pytest.fixture(scope="module")
def dask_client():
    # start_dask_client will check for the SCHEDULER_FILE and
    # DASK_WORKER_DEVICES env vars and use them when creating a client if
    # set. start_dask_client will also initialize the Comms singleton.
    dask_client, dask_cluster = start_dask_client()

    yield dask_client

    stop_dask_client(dask_client, dask_cluster)


@pytest.fixture(scope="module")
def scratch_dir():
    # This should always be set if doing MG testing, since temporary
    # directories are only accessible from the current process.
    tempdir_object = os.getenv(
        "RAPIDS_PYTEST_SCRATCH_DIR", tempfile.TemporaryDirectory()
    )

    if isinstance(tempdir_object, tempfile.TemporaryDirectory):
        yield tempdir_object.name
    else:
        yield tempdir_object

    del tempdir_object
