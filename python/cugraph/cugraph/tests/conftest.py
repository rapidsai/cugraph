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
from cugraph.dask.common.mg_utils import start_dask_client, teardown_local_dask_cluster

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
    cluster, client = start_dask_client()
    yield client

    teardown_local_dask_cluster(cluster, client)
