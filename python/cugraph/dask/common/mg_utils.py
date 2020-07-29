# Copyright (c) 2020, NVIDIA CORPORATION.
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
from cugraph.raft.dask.common.utils import default_client
from collections import OrderedDict


def is_worker_organizer(worker_idx):
    return worker_idx == 0


# FIXME: We currently look for the default client from dask, as such is the
# if there is a dask client running without any GPU we will still try
# to run OPG using this client, it also implies that more  work will be
# required  in order to run an OPG Batch in Combination with mutli-GPU Graph
def mg_get_client():
    try:
        client = default_client()
    except ValueError:
        client = None
    return client


def mg_get_worker_addresses(client=None):
    if client is None:
        client = mg_get_client
    addresses = list(OrderedDict.fromkeys(
                     client.scheduler_info()["workers"].keys()))
    return addresses


def mg_prepare_worker_to_parts(data, client=None):
    if client is None:
        client = mg_get_client
    for placeholder, worker in enumerate(client.has_what().keys()):
        if worker not in data.worker_to_parts:
            data.worker_to_parts[worker] = [placeholder]
    return data
