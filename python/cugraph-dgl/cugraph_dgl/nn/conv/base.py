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

from typing import Optional, Tuple

from cugraph.utilities.utils import import_optional

torch = import_optional("torch")

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
    r"""A god-class to store different sparse formats needed by cugraph-ops
    and facilitate sparse format conversions.

    Parameters
    ----------
    size: tuple of int
        Size of the adjacency matrix: (num_src_nodes, num_dst_nodes).

    src_ids: torch.Tensor
        Source indices of the edges.

    dst_ids: torch.Tensor, optional
        Destination indices of the edges.

    csrc_ids: torch.Tensor, optional
        Compressed source indices. It is a monotonically increasing array of
        size (num_src_nodes + 1,). For the k-th source node, its neighborhood
        consists of the destinations between `dst_indices[csrc_indices[k]]` and
        `dst_indices[csrc_indices[k+1]]`.

    cdst_ids: torch.Tensor, optional
        Compressed destination indices. It is a monotonically increasing array of
        size (num_dst_nodes + 1,). For the k-th destination node, its neighborhood
        consists of the sources between `src_indices[cdst_indices[k]]` and
        `src_indices[cdst_indices[k+1]]`.

    dst_ids_is_sorted: bool
        Whether `dst_ids` has been sorted in an ascending order.

    Notes
    -----
    COO-format requires `src_ids` and `dst_ids`.
    CSC-format requires `cdst_ids` and `src_ids`.
    CSR-format requires `csrc_ids` and `dst_ids`.

    For MFGs (sampled graphs), the node ids must have been renumbered.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        src_ids: torch.Tensor,
        dst_ids: Optional[torch.Tensor] = None,
        csrc_ids: Optional[torch.Tensor] = None,
        cdst_ids: Optional[torch.Tensor] = None,
        dst_ids_is_sorted: bool = False,
    ):
        if dst_ids is None and cdst_ids is None:
            raise ValueError("One of 'dst_ids' and 'cdst_ids' must be given.")

        if src_ids is not None:
            src_ids = src_ids.contiguous()
        if dst_ids is not None:
            dst_ids = dst_ids.contiguous()
        if csrc_ids is not None:
            csrc_ids = csrc_ids.contiguous()
        if cdst_ids is not None:
            cdst_ids = cdst_ids.contiguous()

        self._src_ids = src_ids
        self._dst_ids = dst_ids
        self._csrc_ids = csrc_ids
        self._cdst_ids = cdst_ids
        self.num_src_nodes, self.num_dst_nodes = size

        # Force create CSC format.
        if self._cdst_ids is None:
            if not dst_ids_is_sorted:
                self._dst_ids, self._perm = torch.sort(self._dst_ids)
                self._src_ids = self._src_ids[self._perm]
            self._cdst_ids = torch._convert_indices_from_coo_to_csr(
                self._dst_ids,
                self.num_dst_nodes,
                out_int32=self._dst_ids.dtype == torch.int32,
            )

    def csc(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return CSC format."""
        return (self._cdst_ids, self._src_ids)
