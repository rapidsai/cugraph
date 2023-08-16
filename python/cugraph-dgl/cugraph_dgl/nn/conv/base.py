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

from typing import Optional, Tuple, Union

from cugraph.utilities.utils import import_optional

# torch = import_optional("torch")
import torch

ops_torch = import_optional("pylibcugraphops.pytorch")


class BaseConv(torch.nn.Module):
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


def compress_ids(ids: torch.Tensor, size: int) -> torch.Tensor:
    return torch._convert_indices_from_coo_to_csr(
        ids, size, out_int32=ids.dtype == torch.int32
    )


def decompress_ids(c_ids: torch.Tensor) -> torch.Tensor:
    ids = torch.arange(c_ids.numel() - 1, dtype=c_ids.dtype, device=c_ids.device)
    return ids.repeat_interleave(c_ids[1:] - c_ids[:-1])


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
        Whether `dst_ids` has been sorted in an ascending order. When sorted,
        creating CSC layout is much faster.

    formats: str or tuple of str, optional
        The desired sparse formats to create for the graph.

    reduce_memory: bool, optional
        When set, the tensors are not required by the desired formats will be
        set to `None`.

    Notes
    -----
    For MFGs (sampled graphs), the node ids must have been renumbered.
    """

    supported_formats = {"coo": ("src_ids", "dst_ids"), "csc": ("cdst_ids", "src_ids")}

    all_tensors = set(["src_ids", "dst_ids", "csrc_ids", "cdst_ids"])

    def __init__(
        self,
        size: Tuple[int, int],
        src_ids: torch.Tensor,
        dst_ids: Optional[torch.Tensor] = None,
        csrc_ids: Optional[torch.Tensor] = None,
        cdst_ids: Optional[torch.Tensor] = None,
        dst_ids_is_sorted: bool = False,
        formats: Optional[Union[str, Tuple[str]]] = None,
        reduce_memory: bool = True,
    ):
        self._num_src_nodes, self._num_dst_nodes = size
        self._dst_ids_is_sorted = dst_ids_is_sorted

        if dst_ids is None and cdst_ids is None:
            raise ValueError("One of 'dst_ids' and 'cdst_ids' must be given.")

        if src_ids is not None:
            src_ids = src_ids.contiguous()

        if dst_ids is not None:
            dst_ids = dst_ids.contiguous()

        if csrc_ids is not None:
            if csrc_ids.numel() != self._num_src_nodes + 1:
                raise RuntimeError(
                    f"Size mismatch for 'csrc_ids': expected ({size[0]+1},), "
                    f"but got {tuple(csrc_ids.size())}"
                )
            csrc_ids = csrc_ids.contiguous()

        if cdst_ids is not None:
            if cdst_ids.numel() != self._num_dst_nodes + 1:
                raise RuntimeError(
                    f"Size mismatch for 'cdst_ids': expected ({size[1]+1},), "
                    f"but got {tuple(cdst_ids.size())}"
                )
            cdst_ids = cdst_ids.contiguous()

        self._src_ids = src_ids
        self._dst_ids = dst_ids
        self._csrc_ids = csrc_ids
        self._cdst_ids = cdst_ids
        self._perm = None

        if isinstance(formats, str):
            formats = (formats,)

        if formats is not None:
            for format_ in formats:
                assert format_ in SparseGraph.supported_formats
                self.__getattribute__(f"_create_{format_}")()
        self._formats = formats

        self._reduce_memory = reduce_memory
        if reduce_memory:
            self.reduce_memory()

    def reduce_memory(self):
        """Remove the tensors that are not necessary to create the desired sparse
        formats to reduce memory footprint."""

        self._perm = None
        if self._formats is None:
            return

        tensors_needed = []
        for f in self._formats:
            tensors_needed += SparseGraph.supported_formats[f]
        for t in SparseGraph.all_tensors.difference(set(tensors_needed)):
            self.__dict__[t] = None

    def _create_coo(self):
        if self._dst_ids is None:
            self._dst_ids = decompress_ids(self._cdst_ids)

    def _create_csc(self):
        if self._cdst_ids is None:
            if not self._dst_ids_is_sorted:
                self._dst_ids, self._perm = torch.sort(self._dst_ids)
                self._src_ids = self._src_ids[self._perm]
            self._cdst_ids = compress_ids(self._dst_ids, self._num_dst_nodes)

    def num_src_nodes(self):
        return self._num_src_nodes

    def num_dst_nodes(self):
        return self._num_dst_nodes

    def formats(self):
        return self._formats

    def coo(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if "coo" not in self.formats():
            raise RuntimeError(
                "The SparseGraph did not create a COO layout. "
                "Set 'formats' to include 'coo' when creating the graph."
            )
        return (self._src_ids, self._dst_ids)

    def csc(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if "csc" not in self.formats():
            raise RuntimeError(
                "The SparseGraph did not create a CSC layout. "
                "Set 'formats' to include 'csc' when creating the graph."
            )
        return (self._cdst_ids, self._src_ids)
