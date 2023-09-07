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
from pylibcugraphops.pytorch.operators import agg_hg_basis_n2n_post

from .base import BaseConv

torch = import_optional("torch")
torch_geometric = import_optional("torch_geometric")


class RGCNConv(BaseConv):  # pragma: no cover
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int, optional): If set, this layer will use the
            basis-decomposition regularization scheme where :obj:`num_bases`
            denotes the number of bases to use. (default: :obj:`None`)
        aggr (str, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"sum"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        aggr: str = "mean",
        root_weight: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        if aggr not in ["mean", "sum", "add"]:
            raise ValueError(
                f"Aggregation function must be chosen from 'mean', 'sum' or "
                f"'add', but got '{aggr}'."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.aggr = aggr
        self.root_weight = root_weight

        dim_root_weight = 1 if root_weight else 0

        if num_bases is not None:
            self.weight = torch.nn.Parameter(
                torch.empty(num_bases + dim_root_weight, in_channels, out_channels)
            )
            self.comp = torch.nn.Parameter(torch.empty(num_relations, num_bases))
        else:
            self.weight = torch.nn.Parameter(
                torch.empty(num_relations + dim_root_weight, in_channels, out_channels)
            )
            self.register_parameter("comp", None)

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        end = -1 if self.root_weight else None
        torch_geometric.nn.inits.glorot(self.weight[:end])
        torch_geometric.nn.inits.glorot(self.comp)
        if self.root_weight:
            torch_geometric.nn.inits.glorot(self.weight[-1])
        torch_geometric.nn.inits.zeros(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        csc: Tuple[torch.Tensor, torch.Tensor, int],
        edge_type: torch.Tensor,
        max_num_neighbors: Optional[int] = None,
    ) -> torch.Tensor:

        graph = self.get_typed_cugraph(
            csc, edge_type, self.num_relations, max_num_neighbors=max_num_neighbors
        )

        out = agg_hg_basis_n2n_post(
            x,
            self.comp,
            graph,
            concat_own=self.root_weight,
            norm_by_out_degree=bool(self.aggr == "mean"),
        )

        out = out @ self.weight.view(-1, self.out_channels)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, num_relations={self.num_relations})"
        )
