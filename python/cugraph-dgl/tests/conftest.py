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

import dgl
import torch

from cugraph.testing.mg_utils import (
    start_dask_client,
    stop_dask_client,
)


@pytest.fixture(scope="module")
def dask_client():
    # start_dask_client will check for the SCHEDULER_FILE and
    # DASK_WORKER_DEVICES env vars and use them when creating a client if
    # set. start_dask_client will also initialize the Comms singleton.
    dask_client, dask_cluster = start_dask_client(
        dask_worker_devices="0", protocol="tcp"
    )

    yield dask_client

    stop_dask_client(dask_client, dask_cluster)


class SparseGraphData1:
    size = (6, 5)
    nnz = 6
    src_ids = torch.IntTensor([0, 1, 2, 3, 2, 5]).cuda()
    dst_ids = torch.IntTensor([1, 2, 3, 4, 0, 3]).cuda()
    values = torch.IntTensor([10, 20, 30, 40, 50, 60]).cuda()

    # CSR
    src_ids_sorted_by_src = torch.IntTensor([0, 1, 2, 2, 3, 5]).cuda()
    dst_ids_sorted_by_src = torch.IntTensor([1, 2, 0, 3, 4, 3]).cuda()
    csrc_ids = torch.IntTensor([0, 1, 2, 4, 5, 5, 6]).cuda()
    values_csr = torch.IntTensor([10, 20, 50, 30, 40, 60]).cuda()

    # CSC
    src_ids_sorted_by_dst = torch.IntTensor([2, 0, 1, 5, 2, 3]).cuda()
    dst_ids_sorted_by_dst = torch.IntTensor([0, 1, 2, 3, 3, 4]).cuda()
    cdst_ids = torch.IntTensor([0, 1, 2, 3, 5, 6]).cuda()
    values_csc = torch.IntTensor([50, 10, 20, 60, 30, 40]).cuda()


@pytest.fixture
def sparse_graph_1():
    return SparseGraphData1()


@pytest.fixture
def dgl_graph_1():
    src = torch.tensor([0, 1, 0, 2, 3, 0, 4, 0, 5, 0, 6, 7, 0, 8, 9])
    dst = torch.tensor([1, 9, 2, 9, 9, 4, 9, 5, 9, 6, 9, 9, 8, 9, 0])
    return dgl.graph((src, dst))
