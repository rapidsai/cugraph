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
"""Torch Module for graph attention network layer using the aggregation
primitives in cugraph-ops"""
# pylint: disable=no-member, arguments-differ, invalid-name, too-many-arguments
from __future__ import annotations
from typing import Optional, Tuple, Union

from cugraph_dgl.nn.conv.base import BaseConv
from cugraph.utilities.utils import import_optional

dgl = import_optional("dgl")
torch = import_optional("torch")
nn = import_optional("torch.nn")
ops_torch = import_optional("pylibcugraphops.pytorch")


class GATConv(BaseConv):
    r"""Graph attention layer from `Graph Attention Network
    <https://arxiv.org/pdf/1710.10903.pdf>`__, with the sparse aggregation
    accelerated by cugraph-ops.

    Parameters
    ----------
    in_feats : int, pair of ints
        Input feature size. A pair denotes feature sizes of source and
        destination nodes.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    concat : bool, optional
        If False, the multi-head attentions are averaged instead of concatenated.
        Default: ``True``.
    edge_feats : int, optional
        Edge feature size. Default: ``None``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    bias : bool, optional
        If True, learns a bias term. Defaults: ``True``.

    Examples
    --------
    >>> import dgl
    >>> import torch
    >>> from cugraph_dgl.nn import GATConv
    ...
    >>> device = 'cuda'
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3])).to(device)
    >>> g = dgl.add_self_loop(g)
    >>> feat = torch.ones(6, 10).to(device)
    >>> conv = GATConv(10, 2, num_heads=3).to(device)
    >>> res = conv(g, feat)
    >>> res
    tensor([[[ 0.2340,  1.9226],
            [ 1.6477, -1.9986],
            [ 1.1138, -1.9302]],
            [[ 0.2340,  1.9226],
            [ 1.6477, -1.9986],
            [ 1.1138, -1.9302]],
            [[ 0.2340,  1.9226],
            [ 1.6477, -1.9986],
            [ 1.1138, -1.9302]],
            [[ 0.2340,  1.9226],
            [ 1.6477, -1.9986],
            [ 1.1138, -1.9302]],
            [[ 0.2340,  1.9226],
            [ 1.6477, -1.9986],
            [ 1.1138, -1.9302]],
            [[ 0.2340,  1.9226],
            [ 1.6477, -1.9986],
            [ 1.1138, -1.9302]]], device='cuda:0', grad_fn=<ViewBackward0>)
    """
    MAX_IN_DEGREE_MFG = 200

    def __init__(
        self,
        in_feats: Union[int, Tuple[int, int]],
        out_feats: int,
        num_heads: int,
        concat: bool = True,
        edge_feats: Optional[int] = None,
        negative_slope: float = 0.2,
        bias: bool = True,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.concat = concat
        self.edge_feats = edge_feats
        self.negative_slope = negative_slope

        if isinstance(in_feats, int):
            self.fc = nn.Linear(in_feats, num_heads * out_feats, bias=False)
        else:
            self.fc_src = nn.Linear(in_feats[0], num_heads * out_feats, bias=False)
            self.fc_dst = nn.Linear(in_feats[1], num_heads * out_feats, bias=False)

        if edge_feats is not None:
            self.fc_edge = nn.Linear(edge_feats, num_heads * out_feats, bias=False)
            self.attn_weights = nn.Parameter(torch.Tensor(3 * num_heads * out_feats))
        else:
            self.register_parameter("fc_edge", None)
            self.attn_weights = nn.Parameter(torch.Tensor(2 * num_heads * out_feats))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_heads, out_feats))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)

        nn.init.xavier_normal_(
            self.attn_weights.view(-1, self.num_heads, self.out_feats), gain=gain
        )
        if self.fc_edge is not None:
            self.fc_edge.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        g: dgl.DGLHeteroGraph,
        nfeat: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        efeat: Optional[torch.Tensor] = None,
        max_in_degree: Optional[int] = None,
    ) -> torch.Tensor:
        r"""Forward computation.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        nfeat : torch.Tensor
            Input features of shape :math:`(N, D_{in})`.
        efeat: torch.Tensor, optional
            Optional edge features.
        max_in_degree : int
            Maximum in-degree of destination nodes. It is only effective when
            :attr:`g` is a :class:`DGLBlock`, i.e., bipartite graph. When
            :attr:`g` is generated from a neighbor sampler, the value should be
            set to the corresponding :attr:`fanout`. If not given,
            :attr:`max_in_degree` will be calculated on-the-fly.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where
            :math:`H` is the number of heads, and :math:`D_{out}` is size of
            output feature.
        """
        if max_in_degree is None:
            max_in_degree = -1

        bipartite = not isinstance(nfeat, torch.Tensor)
        offsets, indices, _ = g.adj_tensors("csc")

        graph = ops_torch.CSC(
            offsets=offsets,
            indices=indices,
            num_src_nodes=g.num_src_nodes(),
            dst_max_in_degree=max_in_degree,
            is_bipartite=bipartite,
        )

        if efeat is not None:
            if self.fc_edge is None:
                raise RuntimeError(
                    f"{self.__class__.__name__}.edge_feats must be set to "
                    f"accept edge features."
                )
            efeat = self.fc_edge(efeat)

        if bipartite:
            if not hasattr(self, "fc_src"):
                raise RuntimeError(
                    f"{self.__class__.__name__}.in_feats must be a pair of "
                    f"integers to allow bipartite node features, but got "
                    f"{self.in_feats}."
                )
            nfeat_src = self.fc_src(nfeat[0])
            nfeat_dst = self.fc_dst(nfeat[1])
        else:
            if not hasattr(self, "fc"):
                raise RuntimeError(
                    f"{self.__class__.__name__}.in_feats is expected to be an "
                    f"integer, but got {self.in_feats}."
                )
            nfeat = self.fc(nfeat)

        out = ops_torch.operators.mha_gat_n2n(
            (nfeat_src, nfeat_dst) if bipartite else nfeat,
            self.attn_weights,
            graph,
            num_heads=self.num_heads,
            activation="LeakyReLU",
            negative_slope=self.negative_slope,
            concat_heads=self.concat,
            edge_feat=efeat,
        )[: g.num_dst_nodes()]

        if self.concat:
            out = out.view(-1, self.num_heads, self.out_feats)

        if self.bias is not None:
            out = out + self.bias

        return out
