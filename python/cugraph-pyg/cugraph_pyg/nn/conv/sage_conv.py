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
from pylibcugraphops.pytorch.operators import agg_concat_n2n

from .base import BaseConv

torch = import_optional("torch")
torch_geometric = import_optional("torch_geometric")


class SAGEConv(BaseConv):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    If :obj:`project = True`, then :math:`\mathbf{x}_j` will first get
    projected via

    .. math::
        \mathbf{x}_j \leftarrow \sigma ( \mathbf{W}_3 \mathbf{x}_j +
        \mathbf{b})

    as described in Eq. (3) of the paper.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        aggr (str or Aggregation, optional): The aggregation scheme to use.
            Choose from :obj:`"mean"`, :obj:`"sum"`, :obj:`"min"` or
            :obj:`"max"`. (default: :obj:`"mean"`)
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{h}_i^{k+1}}
            {\| \mathbf{h}_i^{k+1} \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        project (bool, optional): If set to :obj:`True`, the layer will apply a
            linear transformation followed by an activation function before
            aggregation (as described in Eq. (3) of the paper).
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: str = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        if aggr not in ["mean", "sum", "min", "max"]:
            raise ValueError(
                f"Aggregation function must be chosen from 'mean',"
                f" 'sum', 'min' or 'max', but got '{aggr}'."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            self.in_channels_src = self.in_channels_dst = in_channels
        else:
            self.in_channels_src, self.in_channels_dst = in_channels

        if self.project:
            self.pre_lin = torch_geometric.nn.Linear(
                self.in_channels_src, self.in_channels_src, bias=True
            )

        if self.root_weight:
            self.lin = torch_geometric.nn.Linear(
                self.in_channels_src + self.in_channels_dst, out_channels, bias=bias
            )
        else:
            self.lin = torch_geometric.nn.Linear(
                self.in_channels_src, out_channels, bias=bias
            )

        self.reset_parameters()

    def reset_parameters(self):
        if self.project:
            self.pre_lin.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        csc: Tuple[torch.Tensor, torch.Tensor, int],
        max_num_neighbors: Optional[int] = None,
    ) -> torch.Tensor:
        bipartite = isinstance(x, Tuple)
        graph = self.get_cugraph(
            csc, bipartite=bipartite, max_num_neighbors=max_num_neighbors
        )

        if self.project:
            if bipartite:
                x = (self.pre_lin(x[0]).relu(), x[1])
            else:
                x = self.pre_lin(x).relu()

        out = agg_concat_n2n(x, graph, self.aggr)

        if self.root_weight:
            out = self.lin(out)
        else:
            out = self.lin(out[:, : self.in_channels_src])

        if self.normalize:
            out = torch.nn.functional.normalize(out, p=2.0, dim=-1)

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, aggr={self.aggr})"
        )
