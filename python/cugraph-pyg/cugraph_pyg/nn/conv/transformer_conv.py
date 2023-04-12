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

from pylibcugraphops.pytorch.operators import mha_simple_n2n as TransformerConvAgg

from cugraph.utilities.utils import import_optional

from .base import BaseConv

torch = import_optional("torch")
nn = import_optional("torch.nn")


class TransformerConv(BaseConv):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        edge_dim: Optional[int] = None,
        root_weight: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.root_weight = root_weight
        self.concat = concat
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)

        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                csc: Tuple[torch.Tensor, torch.Tensor, int],
                edge_attr: Optional[torch.Tensor] = None,
                max_num_neighbors: Optional[int] = None):
        graph = self.get_cugraph(csc, max_num_neighbors)

        if isinstance(x, torch.Tensor):
            x = (x, x)

        query = self.lin_query(x[1])
        key = self.lin_key(x[0])
        value = self.lin_value(x[0])

        out = TransformerConvAgg(key, query, value, graph, self.heads, self.concat,
                                 edge_emb=edge_attr, norm_by_dim=False, score_bias=None)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
