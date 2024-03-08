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

from typing import Optional, Sequence, Union

import torch
from torch import nn
import e3nn
from e3nn import o3

from cugraph_equivariant.utils import scatter_reduce

from pylibcugraphops.pytorch.operators import FusedTensorProduct


class MACETensorProductConv(nn.Module):
    r"""MACE module"""

    def __init__(
        self,
        in_irreps: o3.Irreps,
        sh_irreps: o3.Irreps,
        out_irreps: o3.Irreps,
        nu: int,
        mlp_channels: Optional[list[int]] = None,
        # mlp_activation: Union[nn.Module, Sequence[nn.Module]] = nn.GELU(),
        e3nn_compat_mode: bool = False,
    ):
        super().__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.nu = nu
        self.e3nn_compat_mode = e3nn_compat_mode

        instructions = o3.FullTensorProduct(
            in_irreps, sh_irreps, filter_ir_out=out_irreps
        ).instructions
        instructions = [instr._replace(has_weight=True) for instr in instructions]

        self.tp = FusedTensorProduct(
            instructions,
            in_irreps,
            sh_irreps,
            out_irreps,
            e3nn_compat_mode=e3nn_compat_mode,
        )

        self.mlp = e3nn.nn.FullyConnectedNet(
            list(mlp_channels) + [self.tp.weight_numel],
            torch.nn.functional.silu,
        )

    def forward(
        self,
        src_features: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_emb: torch.Tensor,
        graph: tuple[torch.Tensor, tuple[int, int]],
    ):
        (src, dst), (num_src_nodes, num_dst_nodes) = graph

        tp_weights = self.mlp(edge_emb)
        out = self.tp(src_features[src], edge_sh, tp_weights)

        out = scatter_reduce(out, dst, dim=0, dim_size=num_dst_nodes, reduce="sum")

        out = SymmetricContraction(self.nu, self.out_irreps)(out)
        return out


class SymmetricContraction(nn.Module):
    pass
