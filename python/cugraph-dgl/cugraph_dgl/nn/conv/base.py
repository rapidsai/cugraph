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

torch = import_optional("torch")
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
    r"""A class to store different sparse formats needed by cugraph-ops
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

    values: torch.Tensor, optional
        Values on the edges.

    is_sorted: bool
        Whether the COO inputs (src_ids, dst_ids, values) have been sorted by
        `dst_ids` in an ascending order. CSC layout creation is much faster
        when sorted.

    formats: str or tuple of str, optional
        The desired sparse formats to create for the graph. Default: "csc".
        Choose from: ["coo", "csc"], ["csc", "csr"]

    reduce_memory: bool, optional
        When set, the tensors are not required by the desired formats will be
        set to `None`.

    Notes
    -----
    For MFGs (sampled graphs), the node ids must have been renumbered.
    """

    supported_formats = {
        "coo": ("src_ids", "dst_ids"),
        "csc": ("cdst_ids", "src_ids"),
        "csr": ("csrc_ids", "dst_ids", "_perm_csc2csr"),
    }

    all_tensors = set(
        ["src_ids", "dst_ids", "csrc_ids", "cdst_ids", "_perm_coo2csc", "_perm_csc2csr"]
    )

    def __init__(
        self,
        size: Tuple[int, int],
        src_ids: torch.Tensor,
        dst_ids: Optional[torch.Tensor] = None,
        csrc_ids: Optional[torch.Tensor] = None,
        cdst_ids: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        is_sorted: bool = False,
        formats: Union[str, Tuple[str]] = "csc",
        reduce_memory: bool = False,
    ):
        self._num_src_nodes, self._num_dst_nodes = size
        self._is_sorted = is_sorted

        if dst_ids is None and cdst_ids is None:
            raise ValueError(
                "One of 'dst_ids' and 'cdst_ids' must be given "
                "to create a SparseGraph."
            )

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

        if values is not None:
            values = values.contiguous()

        self._src_ids = src_ids
        self._dst_ids = dst_ids
        self._csrc_ids = csrc_ids
        self._cdst_ids = cdst_ids
        self._values = values
        self._perm_coo2csc = None
        self._perm_csc2csr = None

        if isinstance(formats, str):
            formats = (formats,)
        self._formats = formats

        if "csc" not in formats:
            raise ValueError(
                f"{self.__class__.__name__}.formats must contain "
                f"'csc', but got {formats}."
            )

        # always create csc first
        if self._cdst_ids is None:
            if not self._is_sorted:
                self._dst_ids, self._perm_coo2csc = torch.sort(self._dst_ids)
                self._src_ids = self._src_ids[self._perm_coo2csc]
                if self._values is not None:
                    self._values = self._values[self._perm_coo2csc]
            self._cdst_ids = compress_ids(self._dst_ids, self._num_dst_nodes)

        for format_ in formats:
            assert format_ in SparseGraph.supported_formats
            self.__getattribute__(f"{format_}")()

        self._reduce_memory = reduce_memory
        if reduce_memory:
            self.reduce_memory()

    def reduce_memory(self):
        """Remove the tensors that are not necessary to create the desired sparse
        formats to reduce memory footprint."""

        self._perm_coo2csc = None
        if self._formats is None:
            return

        tensors_needed = []
        for f in self._formats:
            tensors_needed += SparseGraph.supported_formats[f]
        for t in SparseGraph.all_tensors.difference(set(tensors_needed)):
            self.__dict__[t] = None

    def src_ids(self) -> torch.Tensor:
        return self._src_ids

    def cdst_ids(self) -> torch.Tensor:
        return self._cdst_ids

    def dst_ids(self) -> torch.Tensor:
        if self._dst_ids is None:
            self._dst_ids = decompress_ids(self._cdst_ids)
        return self._dst_ids

    def csrc_ids(self) -> torch.Tensor:
        if self._csrc_ids is None:
            src_ids, self._perm_csc2csr = torch.sort(self._src_ids)
            self._csrc_ids = compress_ids(src_ids, self._num_src_nodes)
        return self._csrc_ids

    def num_src_nodes(self):
        return self._num_src_nodes

    def num_dst_nodes(self):
        return self._num_dst_nodes

    def formats(self):
        return self._formats

    def coo(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if "coo" not in self.formats():
            raise RuntimeError(
                "The SparseGraph did not create a COO layout. "
                "Set 'formats' list to include 'coo' when creating the graph."
            )
        return self.src_ids(), self.dst_ids(), self._values

    def csc(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if "csc" not in self.formats():
            raise RuntimeError(
                "The SparseGraph did not create a CSC layout. "
                "Set 'formats' list to include 'csc' when creating the graph."
            )
        return self.cdst_ids(), self.src_ids(), self._values

    def csr(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if "csr" not in self.formats():
            raise RuntimeError(
                "The SparseGraph did not create a CSR layout. "
                "Set 'formats' list to include 'csr' when creating the graph."
            )
        csrc_ids = self.csrc_ids()
        dst_ids = self.dst_ids()[self._perm_csc2csr]
        value = self._values
        if value is not None:
            value = value[self._perm_csc2csr]
        return csrc_ids, dst_ids, value
