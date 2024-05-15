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

from __future__ import annotations
from typing import Sequence

import torch
from torch import Tensor
import e3nn
from e3nn import o3
from pylibcugraphops.pytorch.operators import (
    FusedLinear,
    FusedFullyConnectedTensorProduct,
    FusedSymmetricContraction,
)
from .tensor_product_conv import TensorProductConv, Graph
from ._mace_utils import tp_out_irreps_with_instructions


class InteractionBlock(torch.nn.Module):
    """Identical to RealAgnosticInteractionBlock from MACE repo."""

    def __init__(
        self,
        in_irreps: o3.Irreps,
        sh_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        num_elements: o3.Irreps,
        num_bessel_basis: int,
        avg_num_neighbors: float,
        radial_mlp_channels: Sequence[int] | None = None,
        e3nn_compat_mode: bool = False,
    ) -> None:
        super().__init__()
        self.in_irreps = in_irreps
        self.sh_irreps = sh_irreps
        self.target_irreps = target_irreps
        self.num_elements = num_elements
        self.num_bessel_basis = num_bessel_basis
        self.avg_num_neighbors = avg_num_neighbors
        self.e3nn_compat_mode = e3nn_compat_mode

        if radial_mlp_channels is None:
            radial_mlp_channels = [64, 64, 64]
        self.radial_mlp_channels = list(radial_mlp_channels)

        self.linear_up = FusedLinear(
            self.in_irreps,
            self.in_irreps,
            e3nn_compat_mode=self.e3nn_compat_mode,
            internal_weights=True,
            shared_weights=True,
        )

        # generate uvu instructions here
        tp_out_irreps, instructions = tp_out_irreps_with_instructions(
            in_irreps, sh_irreps, target_irreps
        )

        self.tp_conv = TensorProductConv(
            in_irreps,
            sh_irreps,
            tp_out_irreps,
            instructions,
            e3nn_compat_mode=self.e3nn_compat_mode,
        )

        self.radial_mlp = e3nn.nn.FullyConnectedNet(
            [num_bessel_basis]
            + self.radial_mlp_channels
            + [self.tp_conv.tp.weight_numel],
            torch.nn.SiLU(),
        )

        self.linear = FusedLinear(
            tp_out_irreps,
            self.target_irreps,
            e3nn_compat_mode=self.e3nn_compat_mode,
            internal_weights=True,
            shared_weights=True,
        )

        self.skip_tp_conv = FusedFullyConnectedTensorProduct(
            self.target_irreps,
            o3.Irreps(f"{num_elements}x0e"),
            self.target_irreps,
            e3nn_compat_mode=self.e3nn_compat_mode,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
        self,
        node_feats: Tensor,  # (num_nodes, in_irreps.dim)
        node_attrs: Tensor,  # (num_nodes, num_elements)
        edge_feats: Tensor,  # (num_edges, num_bessel_basis)
        edge_attrs: Tensor,  # (num_edges, sh_irreps.dim)
        graph: Graph,
    ):
        node_feats = self.linear_up(node_feats)
        tp_weights = self.radial_mlp(edge_feats)
        message = self.tp_conv(
            src_features=node_feats,
            edge_sh=edge_attrs,
            graph=graph,
            tp_weights=tp_weights,
            reduce="sum",
        )
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp_conv(message, node_attrs)
        return message  # (num_dst_nodes, target_irreps.dim)


class EquivariantProductBasisBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: int,
        num_elements: int,
        e3nn_compat_mode: bool = False,
    ) -> None:
        super().__init__()
        self.e3nn_compat_mode = e3nn_compat_mode
        self.symmetric_contraction = FusedSymmetricContraction(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
        )

        self.linear = FusedLinear(
            target_irreps,
            target_irreps,
            e3nn_compat_mode=self.e3nn_compat_mode,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(self, node_feats: Tensor, node_attrs: Tensor):
        node_feats = self.symmetric_contraction(node_feats, node_attrs)
        return self.linear(node_feats)
