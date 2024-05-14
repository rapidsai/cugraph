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

import torch
from torch_geometric import EdgeIndex


def get_random_edge_index(
    num_src_nodes,
    num_dst_nodes,
    num_edges,
    dtype=None,
    device=None,
):
    row = torch.randint(num_src_nodes, (num_edges,), dtype=dtype, device=device)
    col = torch.randint(num_dst_nodes, (num_edges,), dtype=dtype, device=device)
    edge_index = torch.stack([row, col], dim=0)

    return EdgeIndex(edge_index, sparse_size=(num_src_nodes, num_dst_nodes))
