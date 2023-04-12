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

from pylibcugraphops.pytorch.operators import mha_gat_n2n as GATConvAgg

from cugraph.utilities.utils import import_optional

from .base import BaseConv

torch = import_optional("torch")
nn = import_optional("torch.nn")



class GATConv(BaseConv):  # pragma: no cover
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper.

    :class:`GATConv` is an optimized version of
    :class:`~torch_geometric.nn.conv.GATConv` based on the :obj:`cugraph-ops`
    package that fuses message passing computation for accelerated execution
    and lower memory footprint.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(2 * heads * out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_normal_(
            self.att.view(2, self.heads, self.out_channels), gain=gain
        )
        if self.bias is not None:
            self.bias.data.fill_(0.)

    def forward(
        self,
        x: torch.Tensor,
        csc: Tuple[torch.Tensor, torch.Tensor, int],
        max_num_neighbors: Optional[int] = None,
    ) -> torch.Tensor:
        graph = self.get_cugraph(csc, max_num_neighbors)

        x = self.lin(x)

        out = GATConvAgg(
            x,
            self.att,
            graph,
            self.heads,
            "LeakyReLU",
            self.negative_slope,
            self.concat,
        )

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )
