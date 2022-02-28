# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import numba.cuda

from dask_cuda import LocalCUDACluster
from dask.distributed import Client

from raft.dask.common.utils import default_client
# FIXME: cugraph/__init__.py also imports the comms module, but
# depending on the import environment, cugraph/comms/__init__.py
# may be imported instead. The following imports the comms.py
# module directly
from cugraph.comms import comms as Comms


# FIXME: We currently look for the default client from dask, as such is the
# if there is a dask client running without any GPU we will still try
# to run MG using this client, it also implies that more  work will be
# required  in order to run an MG Batch in Combination with mutli-GPU Graph
def get_client():
    try:
        client = default_client()
    except ValueError:
        client = None
    return client


def prepare_worker_to_parts(data, client=None):
    if client is None:
        client = get_client()
    for placeholder, worker in enumerate(client.has_what().keys()):
        if worker not in data.worker_to_parts:
            data.worker_to_parts[worker] = [placeholder]
    return data


def is_single_gpu():
    ngpus = len(numba.cuda.gpus)
    if ngpus > 1:
        return False
    else:
        return True


def get_visible_devices():
    _visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if _visible_devices is None:
        # FIXME: We assume that if the variable is unset there is only one GPU
        visible_devices = ["0"]
    else:
        visible_devices = _visible_devices.strip().split(",")
    return visible_devices


def setup_local_dask_cluster(p2p=True):
    """
    Performs steps to setup a Dask cluster using LocalCUDACluster and returns
    the LocalCUDACluster and corresponding client instance.
    """
    cluster = LocalCUDACluster()
    client = Client(cluster)
    client.wait_for_workers(len(get_visible_devices()))
    Comms.initialize(p2p=p2p)

    return (cluster, client)


def teardown_local_dask_cluster(cluster, client):
    """
    Performs steps to destroy a Dask cluster and a corresponding client
    instance.
    """
    Comms.destroy()
    client.close()
    cluster.close()
