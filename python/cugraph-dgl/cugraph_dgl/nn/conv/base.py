# Copyright (c) 2023, NVIDIA CORPORATION.
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

from typing import Tuple

from cugraph.utilities.utils import import_optional

from pyg_lib.ops import index_sort

# torch = import_optional("torch")
import torch

nn = import_optional("torch.nn")
ops_torch = import_optional("pylibcugraphops.pytorch")


class BaseConv(nn.Module):
    r"""An abstract base class for cugraph-ops nn module."""

    def __init__(self):
        super().__init__()
        self._cached_offsets_fg = None

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        raise NotImplementedError

    def forward(self, *args):
        r"""Runs the forward pass of the module."""
        raise NotImplementedError

    def pad_offsets(self, offsets: torch.Tensor, size: int) -> torch.Tensor:
        r"""Pad zero-in-degree nodes to the end of offsets to reach size. This
        is used to augment offset tensors from DGL blocks (MFGs) to be
        compatible with cugraph-ops full-graph primitives."""
        if self._cached_offsets_fg is None:
            self._cached_offsets_fg = torch.empty(
                size, dtype=offsets.dtype, device=offsets.device
            )
        elif self._cached_offsets_fg.numel() < size:
            self._cached_offsets_fg.resize_(size)

        self._cached_offsets_fg[: offsets.numel()] = offsets
        self._cached_offsets_fg[offsets.numel() : size] = offsets[-1]

        return self._cached_offsets_fg[:size]


class SparseGraph(object):
    r"""A thin wrapper to facilitate sparse format conversion.

    Parameters
    ----------
    src_ids: torch.Tensor
        Tensor of source indices.
    dst_ids: torch.Tensor
        Tensor of destination indices.
    shape: Tuple of int
        Shape of the adjacency matrix.
    is_sort: bool
        Whether `dst_ids` has been sorted in an ascending order.
    """

    def __init__(
        self,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        shape: Tuple[int, int],
        is_sorted: bool = False,
    ):
        self.row = src_ids.contiguous()
        self.col = dst_ids.contiguous()
        self.num_src_nodes, self.num_dst_nodes = shape
        self.rowptr = None
        self.colptr = None
        self.perm = None

        if not is_sorted:
            self.col, self.perm = index_sort(self.col, max_value=self.num_dst_nodes)
            self.row = self.row[self.perm]

    def to_csc(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.colptr is None:
            self.colptr = torch._convert_indices_from_coo_to_csr(
                self.col, self.num_dst_nodes, out_int32=self.col.dtype == torch.int32
            )

        return (self.row, self.colptr, self.num_src_nodes)
