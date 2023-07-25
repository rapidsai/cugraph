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

from cugraph_dgl.nn.conv.base import BaseConv
from cugraph.utilities.utils import import_optional

dgl = import_optional("dgl")
torch = import_optional("torch")
nn = import_optional("torch.nn")
ops_torch = import_optional("pylibcugraphops.pytorch")


class TransformerConv(BaseConv):
    r"""The graph transformer layer from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper.

    Parameters
    ----------
    in_node_feats : int or pair of ints
        Input feature size. A pair denotes feature sizes of source and
        destination nodes.
    out_node_feats : int
        Output feature size.
    num_heads : int
        Number of multi-head-attentions.
    concat : bool, optional
        If False, the multi-head attentions are averaged instead of concatenated.
        Default: ``True``.
    beta : bool, optional
        If True, use a gated residual connection. Default: ``True``.
    edge_feats: int, optional
        Edge feature size. Default: ``None``.
    bias: bool, optional
        If True, learns a bias term. Default: ``True``.
    root_weight: bool, optional
        If False, will skip to learn a root weight matrix. Default: ``True``.
    """

    def __init__(
        self,
        in_node_feats: Union[int, Tuple[int, int]],
        out_node_feats: int,
        num_heads: int,
        concat: bool = True,
        beta: bool = False,
        edge_feats: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
    ):
        super().__init__()

        self.in_node_feats = in_node_feats
        self.out_node_feats = out_node_feats
        self.num_heads = num_heads
        self.concat = concat
        self.beta = beta
        self.edge_feats = edge_feats
        self.bias = bias
        self.root_weight = root_weight

        if isinstance(in_node_feats, int):
            in_node_feats = (in_node_feats, in_node_feats)

        self.lin_key = nn.Linear(in_node_feats[0], num_heads * out_node_feats)
        self.lin_query = nn.Linear(in_node_feats[1], num_heads * out_node_feats)
        self.lin_value = nn.Linear(in_node_feats[0], num_heads * out_node_feats)

        if edge_feats is not None:
            self.lin_edge = nn.Linear(
                edge_feats, num_heads * out_node_feats, bias=False
            )
        else:
            self.lin_edge = self.register_parameter("lin_edge", None)

        if concat:
            self.lin_skip = nn.Linear(
                in_node_feats[1], num_heads * out_node_feats, bias=bias
            )
            if self.beta:
                self.lin_beta = nn.Linear(3 * num_heads * out_node_feats, 1, bias=bias)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)
        else:
            self.lin_skip = nn.Linear(in_node_feats[1], out_node_feats, bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * out_node_feats, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()
        if self.lin_beta is not None:
            self.lin_beta.reset_parameters()

    def forward(
        self,
        g: dgl.DGLHeteroGraph,
        nfeat: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        efeat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward computation.

        Parameters
        ----------
        g: DGLGraph
            The graph.
        nfeat: torch.Tensor or a pair of torch.Tensor
            Node feature tensor. A pair denotes features for source and
            destination nodes, respectively.
        efeat: torch.Tensor, optional
            Edge feature tensor. Default: ``None``.
        """
        offsets, indices, _ = g.adj_tensors("csc")
        graph = ops_torch.CSC(
            offsets=offsets,
            indices=indices,
            num_src_nodes=g.num_src_nodes(),
            is_bipartite=True,
        )

        if isinstance(nfeat, torch.Tensor):
            nfeat = (nfeat, nfeat)

        query = self.lin_query(nfeat[1][: g.num_dst_nodes()])
        key = self.lin_key(nfeat[0])
        value = self.lin_value(nfeat[0])

        if efeat is not None:
            if self.lin_edge is None:
                raise RuntimeError(
                    f"{self.__class__.__name__}.edge_feats must be set to allow "
                    f"edge features."
                )
            efeat = self.lin_edge(efeat)

        out = ops_torch.operators.mha_simple_n2n(
            key_emb=key,
            query_emb=query,
            value_emb=value,
            graph=graph,
            num_heads=self.num_heads,
            concat_heads=self.concat,
            edge_emb=efeat,
            norm_by_dim=True,
            score_bias=None,
        )[: g.num_dst_nodes()]

        if self.root_weight:
            res = self.lin_skip(nfeat[1][: g.num_dst_nodes()])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, res, out - res], dim=-1))
                beta = beta.sigmoid()
                out = beta * res + (1 - beta) * out
            else:
                out = out + res

        return out
