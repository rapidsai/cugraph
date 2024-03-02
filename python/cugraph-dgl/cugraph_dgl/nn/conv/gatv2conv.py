# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

from typing import Optional, Union

from cugraph_dgl.nn.conv.base import Any, BaseConv, SparseGraph
from cugraph.utilities.utils import import_optional

dgl = import_optional("dgl")
torch = import_optional("torch")
nn = import_optional("torch.nn")
ops_torch = import_optional("pylibcugraphops.pytorch")


class GATv2Conv(BaseConv):
    r"""GATv2 from `How Attentive are Graph Attention Networks?
    <https://arxiv.org/pdf/2105.14491.pdf>`__, with the sparse aggregation
    accelerated by cugraph-ops.

    Parameters
    ----------
    in_feats : int or (int, int)
        Input feature size. A pair denotes feature sizes of source and
        destination nodes.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    concat : bool, optional
        If False, the multi-head attentions are averaged instead of concatenated.
        Default: ``True``.
    edge_feats : int, optional
        Edge feature size. Default: ``None``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will
        be invalid since no message will be passed to those nodes. This is
        harmful for some applications causing silent performance regression.
        This module will raise a DGLError if it detects 0-in-degree nodes in
        input graph. By setting ``True``, it will suppress the check and let the
        users handle it by themselves. Defaults: ``False``.
    bias : bool, optional
        If True, learns a bias term. Defaults: ``True``.
    share_weights : bool, optional
        If ``True``, the same matrix will be applied to the source and the
        destination node features. Defaults: ``False``.
    """

    def __init__(
        self,
        in_feats: Union[int, tuple[int, int]],
        out_feats: int,
        num_heads: int,
        feat_drop: float = 0.0,
        concat: bool = True,
        edge_feats: Optional[int] = None,
        negative_slope: float = 0.2,
        residual: bool = False,
        allow_zero_in_degree: bool = False,
        bias: bool = True,
        share_weights: bool = False,
    ):
        super().__init__()

        if isinstance(in_feats, int):
            self.in_feats_src = self.in_feats_dst = in_feats
        else:
            self.in_feats_src, self.in_feats_dst = in_feats
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.feat_drop = nn.Dropout(feat_drop)
        self.concat = concat
        self.edge_feats = edge_feats
        self.negative_slope = negative_slope
        self.residual = residual
        self.allow_zero_in_degree = allow_zero_in_degree
        self.share_weights = share_weights
        self.bias = bias

        self.lin_src = nn.Linear(self.in_feats_src, num_heads * out_feats, bias=bias)
        if share_weights:
            if self.in_feats_src != self.in_feats_dst:
                raise ValueError(
                    f"Input feature size of source and destination "
                    f"nodes must be identical when share_weights is enabled, "
                    f"but got {self.in_feats_src} and {self.in_feats_dst}."
                )
            self.lin_dst = self.lin_src
        else:
            self.lin_dst = nn.Linear(
                self.in_feats_dst, num_heads * out_feats, bias=bias
            )

        self.attn_weights = nn.Parameter(torch.empty(num_heads * out_feats))

        if edge_feats is not None:
            self.lin_edge = nn.Linear(edge_feats, num_heads * out_feats, bias=False)
        else:
            self.register_parameter("lin_edge", None)

        out_dim = num_heads * out_feats if concat else out_feats
        if residual:
            if self.in_feats_dst != out_dim:
                self.lin_res = nn.Linear(self.in_feats_dst, out_dim, bias=bias)
            else:
                self.lin_res = nn.Identity()
        else:
            self.register_buffer("lin_res", None)

        self.reset_parameters()

    def set_allow_zero_in_degree(self, set_value):
        r"""Set allow_zero_in_degree flag."""
        self.allow_zero_in_degree = set_value

    def reset_parameters(self):
        r"""Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.lin_src.weight, gain=gain)
        nn.init.xavier_normal_(self.lin_dst.weight, gain=gain)

        nn.init.xavier_normal_(
            self.attn_weights.view(-1, self.num_heads, self.out_feats), gain=gain
        )
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()

        if self.lin_res is not None:
            self.lin_res.reset_parameters()

    def forward(
        self,
        g: Union[SparseGraph, dgl.DGLHeteroGraph],
        nfeat: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        efeat: Optional[torch.Tensor] = None,
        max_in_degree: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Forward computation.

        Parameters
        ----------
        graph : DGLGraph or SparseGraph
            The graph.
        nfeat : torch.Tensor
            Input features of shape :math:`(N, D_{in})`.
        efeat: torch.Tensor, optional
            Optional edge features.
        max_in_degree : int
            Maximum in-degree of destination nodes. When :attr:`g` is generated
            from a neighbor sampler, the value should be set to the corresponding
            :attr:`fanout`. This option is used to invoke the MFG-variant of
            cugraph-ops kernel.
        **kwargs : Any
            Additional arguments of `pylibcugraphops.pytorch.operators.mha_gat_v2_n2n`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where
            :math:`H` is the number of heads, and :math:`D_{out}` is size of
            output feature.
        """

        if isinstance(g, dgl.DGLHeteroGraph):
            if not self.allow_zero_in_degree:
                if (g.in_degrees() == 0).any():
                    raise dgl.base.DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

        nfeat_bipartite = isinstance(nfeat, (list, tuple))
        graph_bipartite = nfeat_bipartite or self.share_weights is False

        _graph = self.get_cugraph_ops_CSC(
            g, is_bipartite=graph_bipartite, max_in_degree=max_in_degree
        )
        if kwargs.get("deterministic_dgrad", False):
            _graph.add_reverse_graph()

        if nfeat_bipartite:
            nfeat = (self.feat_drop(nfeat[0]), self.feat_drop(nfeat[1]))
            nfeat_dst_orig = nfeat[1]
        else:
            nfeat = self.feat_drop(nfeat)
            nfeat_dst_orig = nfeat[: g.num_dst_nodes()]

        if efeat is not None:
            if self.lin_edge is None:
                raise RuntimeError(
                    f"{self.__class__.__name__}.edge_feats must be set to "
                    f"accept edge features."
                )
            efeat = self.lin_edge(efeat)

        if nfeat_bipartite:
            nfeat = (self.lin_src(nfeat[0]), self.lin_dst(nfeat[1]))
        elif graph_bipartite:
            nfeat = (self.lin_src(nfeat), self.lin_dst(nfeat[: g.num_dst_nodes()]))
        else:
            nfeat = self.lin_src(nfeat)

        out = ops_torch.operators.mha_gat_v2_n2n(
            nfeat,
            self.attn_weights,
            _graph,
            num_heads=self.num_heads,
            activation="LeakyReLU",
            negative_slope=self.negative_slope,
            concat_heads=self.concat,
            edge_feat=efeat,
            **kwargs,
        )[: g.num_dst_nodes()]

        if self.concat:
            out = out.view(-1, self.num_heads, self.out_feats)

        if self.residual:
            res = self.lin_res(nfeat_dst_orig).view(-1, self.num_heads, self.out_feats)
            if not self.concat:
                res = res.mean(dim=1)
            out = out + res

        return out
