# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import time
import os

from dask.distributed import Client
from dask_cuda import LocalCUDACluster as CUDACluster
import cugraph.comms as Comms
import pytest

# Maximal number of verifications of the number of workers
DEFAULT_MAX_ATTEMPT = 100

# Time between each attempt in seconds
DEFAULT_WAIT_TIME = 0.5


def get_visible_devices():
    _visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if _visible_devices is None:
        # FIXME: We assume that if the variable is unset there is only one GPU
        visible_devices = ["0"]
    else:
        visible_devices = _visible_devices.strip().split(",")
    return visible_devices


def skip_if_not_enough_devices(required_devices):
    visible_devices = get_visible_devices()
    number_of_visible_devices = len(visible_devices)
    if required_devices > number_of_visible_devices:
        pytest.skip("Not enough devices available to "
                    "test MG({})".format(required_devices))


class MGContext:
    """Utility Context Manager to start a multi GPU context using dask_cuda

    Parameters:
    -----------

    number_of_devices : int
        Number of devices to use, verification must be done prior to call to
        ensure that there are enough devices available. If not specified, the
        cluster will be initialized to use all visible devices.
    rmm_managed_memory : bool
        True to enable managed memory (UVM) in RMM as part of the
        cluster. Default is False.
    p2p : bool
        Initialize UCX endpoints if True. Default is False.
    """
    def __init__(self,
                 number_of_devices=None,
                 rmm_managed_memory=False,
                 p2p=False):
        self._number_of_devices = number_of_devices
        self._rmm_managed_memory = rmm_managed_memory
        self._client = None
        self._p2p = p2p
        self._cluster = CUDACluster(
            n_workers=self._number_of_devices,
            rmm_managed_memory=self._rmm_managed_memory
        )

    @property
    def client(self):
        return self._client

    @property
    def cluster(self):
        return self._cluster

    def __enter__(self):
        self._prepare_mg()
        return self

    def _prepare_mg(self):
        self._prepare_client()
        self._prepare_comms()

    def _prepare_client(self):
        self._client = Client(self._cluster)
        self._client.wait_for_workers(self._number_of_devices)

    def _prepare_comms(self):
        Comms.initialize(p2p=self._p2p)

    def _close(self):
        Comms.destroy()
        if self._client is not None:
            self._client.close()
        if self._cluster is not None:
            self._cluster.close()

    def __exit__(self, type, value, traceback):
        self._close()


# NOTE: This only looks for the number of  workers
# Tries to rescale the given cluster and wait until all workers are ready
# or until the maximal number of attempts is reached
def enforce_rescale(cluster, scale, max_attempts=DEFAULT_MAX_ATTEMPT,
                    wait_time=DEFAULT_WAIT_TIME):
    cluster.scale(scale)
    attempt = 0
    ready = (len(cluster.workers) == scale)
    while (attempt < max_attempts) and not ready:
        time.sleep(wait_time)
        ready = (len(cluster.workers) == scale)
        attempt += 1
    assert ready, "Unable to rescale cluster to {}".format(scale)
