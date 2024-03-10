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

from typing import Optional

import torch


def broadcast(src: torch.Tensor, ref: torch.Tensor, dim: int) -> torch.Tensor:
    size = ((1,) * dim) + (-1,) + ((1,) * (ref.dim() - dim - 1))
    return src.view(size).expand_as(ref)


def scatter_reduce(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,  # value of out.size(dim)
    reduce: str = "sum",  # "sum", "prod", "mean", "amax", "amin"
):
    # scatter() expects index to be int64
    index = broadcast(index, src, dim).to(torch.int64)

    size = list(src.size())

    if dim_size is not None:
        size[dim] = dim_size
    else:
        size[dim] = 0 if index.numel() == 0 else int(index.max()) + 1

    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_reduce_(dim, index, src, reduce, include_self=False)
