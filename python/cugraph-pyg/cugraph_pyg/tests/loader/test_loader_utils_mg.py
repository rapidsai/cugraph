# Copyright (c) 2024, NVIDIA CORPORATION.
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

from cugraph.utilities.utils import import_optional, MissingModule


from cugraph.gnn import (
    cugraph_comms_init,
    cugraph_comms_shutdown,
    cugraph_comms_create_unique_id,
)

from cugraph_pyg.loader.loader_utils import scatter

torch = import_optional("torch")
torch_geometric = import_optional("torch_geometric")


def init_pytorch_worker(rank, world_size, cugraph_id):
    import rmm

    rmm.reinitialize(
        devices=rank,
    )

    import cupy

    cupy.cuda.Device(rank).use()
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)

    from cugraph.testing.mg_utils import enable_spilling

    enable_spilling()

    torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    cugraph_comms_init(rank=rank, world_size=world_size, uid=cugraph_id, device=rank)


def run_test_loader_utils_scatter(rank, world_size, uid):
    init_pytorch_worker(rank, world_size, uid)

    num_values_rank = (1 + rank) * 9
    local_values = torch.arange(0, num_values_rank) + 9 * (
        rank + ((rank * (rank - 1)) // 2)
    )

    scatter_perm = torch.tensor_split(torch.arange(local_values.numel()), world_size)

    new_values = scatter(local_values, scatter_perm, rank, world_size)
    print(
        rank,
        local_values,
        new_values,
        flush=True,
    )

    offset = 0
    for send_rank in range(world_size):
        num_values_send_rank = (1 + send_rank) * 9

        expected_values = torch.tensor_split(
            torch.arange(0, num_values_send_rank)
            + 9 * (send_rank + ((send_rank * (send_rank - 1)) // 2)),
            world_size,
        )[rank]

        ix_sent = torch.arange(expected_values.numel())
        values_rec = new_values[ix_sent + offset].cpu()
        offset += values_rec.numel()

        assert (values_rec == expected_values).all()

    cugraph_comms_shutdown()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.mg
def test_loader_utils_scatter():
    uid = cugraph_comms_create_unique_id()
    world_size = torch.cuda.device_count()

    torch.multiprocessing.spawn(
        run_test_loader_utils_scatter,
        args=(world_size, uid),
        nprocs=world_size,
    )
