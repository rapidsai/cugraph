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
from typing import Optional

from cugraph.utilities.utils import import_optional

dgl = import_optional("dgl")
torch = import_optional("torch")
nn = import_optional("torch.nn")
ops = import_optional("pylibcugraphops")
ops_autograd = import_optional("pylibcugraphops.torch.autograd")


class GATConv(nn.Module):
    r"""Graph attention layer from `Graph Attention Network
    <https://arxiv.org/pdf/1710.10903.pdf>`__, with the sparse aggregation
    accelerated by cugraph-ops.

    See :class:`dgl.nn.pytorch.conv.GATConv` for mathematical model.

    This module depends on :code:`pylibcugraphops` package, which can be
    installed via :code:`conda install -c nvidia pylibcugraphops>=23.02`.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
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

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        num_heads: int,
        negative_slope: float = 0.2,
        bias: bool = True,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.negative_slope = negative_slope

        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_weights = nn.Parameter(torch.Tensor(2 * num_heads * out_feats))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_heads * out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Reinitialize learnable parameters."""

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(
            self.attn_weights.view(2, self.num_heads, self.out_feats), gain=gain
        )
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        g: dgl.DGLHeteroGraph,
        feat: torch.Tensor,
        max_in_degree: Optional[int] = None,
    ) -> torch.Tensor:
        r"""Forward computation.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            Input features of shape :math:`(N, D_{in})`.
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

        offsets, indices, _ = g.adj_sparse("csc")

        if g.is_block:
            if max_in_degree is None:
                max_in_degree = g.in_degrees().max().item()
            _graph = ops.make_mfg_csr(
                g.dstnodes(), offsets, indices, max_in_degree, g.num_src_nodes()
            )
        else:
            _graph = ops.make_fg_csr(offsets, indices)

        feat_transformed = self.fc(feat)
        out = ops_autograd.mha_gat_n2n(
            feat_transformed,
            self.attn_weights,
            _graph,
            self.num_heads,
            "LeakyReLU",
            self.negative_slope,
            add_own_node=False,
            concat_heads=True,
        ).view(-1, self.num_heads, self.out_feats)

        if self.bias is not None:
            out = out + self.bias

        return out
