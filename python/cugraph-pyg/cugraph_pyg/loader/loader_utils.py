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

from cugraph.utilities.utils import import_optional
from typing import List

torch = import_optional("torch")


def scatter(
    t: "torch.Tensor", scatter_perm: List["torch.Tensor"], rank: int, world_size: int
):
    """
    t: torch.Tensor
        The local tensor being scattered.
    scatter_perm: List[torch.Tensor]
        The indices to send to each rank.
    rank: int
        The global rank of this worker.
    world_size: int
        The total number of workers.
    """

    scatter_len = torch.tensor(
        [s.numel() for s in scatter_perm], device="cuda", dtype=torch.int64
    )

    scatter_len_all = [
        torch.empty((world_size,), device="cuda", dtype=torch.int64)
        for _ in range(world_size)
    ]
    torch.distributed.all_gather(scatter_len_all, scatter_len)

    t = t.cuda()
    local_tensors = [
        torch.empty((scatter_len_all[r][rank],), device="cuda", dtype=torch.int64)
        for r in range(world_size)
    ]

    qx = []
    for r in range(world_size):
        send_rank = (rank + r) % world_size
        send_op = torch.distributed.P2POp(
            torch.distributed.isend,
            t[scatter_perm[send_rank]],
            send_rank,
        )

        recv_rank = (rank - r) % world_size
        recv_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            local_tensors[recv_rank],
            recv_rank,
        )
        qx += torch.distributed.batch_isend_irecv([send_op, recv_op])

    for x in qx:
        x.wait()

    return torch.concat(local_tensors)
