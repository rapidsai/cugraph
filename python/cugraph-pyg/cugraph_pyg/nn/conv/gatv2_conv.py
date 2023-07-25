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

from .base import BaseConv

torch = import_optional("torch")
nn = import_optional("torch.nn")
torch_geometric = import_optional("torch_geometric")
ops_torch = import_optional("pylibcugraphops.pytorch")


class GATv2Conv(BaseConv):
    r"""The GATv2 operator from the `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the
    static attention problem of the standard
    :class:`~torch_geometric.conv.GATConv` layer.
    Since the linear layers in the standard GAT are applied right after each
    other, the ranking of attended nodes is unconditioned on the query node.
    In contrast, in :class:`GATv2`, every node can attend to any other node.

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k]
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j \, \Vert \, \mathbf{e}_{i,j}]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k \, \Vert \, \mathbf{e}_{i,k}]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        share_weights: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.edge_dim = edge_dim
        self.share_weights = share_weights

        Linear = torch_geometric.nn.Linear

        if isinstance(in_channels, int):
            self.lin_src = Linear(
                in_channels,
                heads * out_channels,
                bias=bias,
                weight_initializer="glorot",
            )

            if share_weights:
                self.lin_dst = self.lin_src
            else:
                self.lin_dst = Linear(
                    in_channels,
                    heads * out_channels,
                    bias=bias,
                    weight_initializer="glorot",
                )
        else:
            self.lin_src = Linear(
                in_channels[0],
                heads * out_channels,
                bias=bias,
                weight_initializer="glorot",
            )
            self.lin_dst = Linear(
                in_channels[1],
                heads * out_channels,
                bias=bias,
                weight_initializer="glorot",
            )

        self.att = nn.Parameter(torch.Tensor(heads * out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(
                edge_dim, heads * out_channels, bias=False, weight_initializer="glorot"
            )
        else:
            self.register_parameter("lin_edge", None)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()

        torch_geometric.nn.inits.glorot(
            self.att.view(-1, self.heads, self.out_channels)
        )

        torch_geometric.nn.inits.zeros(self.bias)

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        csc: Tuple[torch.Tensor, torch.Tensor, int],
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or tuple): The node features. Can be a tuple of
                tensors denoting source and destination node features.
            csc ((torch.Tensor, torch.Tensor, int)): A tuple containing the CSC
                representation of a graph, given as a tuple of
                :obj:`(row, colptr, num_src_nodes)`. Use the
                :meth:`to_csc` method to convert an :obj:`edge_index`
                representation to the desired format.
            edge_attr: (torch.Tensor, optional) The edge features.
        """
        bipartite = not isinstance(x, torch.Tensor) or not self.share_weights
        graph = self.get_cugraph(csc, bipartite=bipartite)

        if edge_attr is not None:
            if self.lin_edge is None:
                raise RuntimeError(
                    f"{self.__class__.__name__}.edge_dim must be set to accept "
                    f"edge features."
                )
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)

        if bipartite:
            if isinstance(x, torch.Tensor):
                x = (x, x)
            x_src = self.lin_src(x[0])
            x_dst = self.lin_dst(x[1])
        else:
            x = self.lin_src(x)

        out = ops_torch.operators.mha_gat_v2_n2n(
            (x_src, x_dst) if bipartite else x,
            self.att,
            graph,
            num_heads=self.heads,
            activation="LeakyReLU",
            negative_slope=self.negative_slope,
            concat_heads=self.concat,
            edge_feat=edge_attr,
        )

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )
