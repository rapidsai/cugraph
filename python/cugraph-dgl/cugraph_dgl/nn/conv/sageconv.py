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

from typing import Optional, Union

from cugraph_dgl.nn.conv.base import BaseConv, SparseGraph
from cugraph.utilities.utils import import_optional

dgl = import_optional("dgl")
torch = import_optional("torch")
nn = import_optional("torch.nn")
ops_torch = import_optional("pylibcugraphops.pytorch")


class SAGEConv(BaseConv):
    r"""An accelerated GraphSAGE layer from `Inductive Representation Learning
    on Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`, with the sparse
    aggregation accelerated by cugraph-ops.

    Parameters
    ----------
    in_feats : int or tuple
        Input feature size. If a scalar is given, the source and destination
        nodes are required to be the same.
    out_feats : int
        Output feature size.
    aggregator_type : str
        Aggregator type to use ("mean", "sum", "min", "max", "pool", "gcn").
    feat_drop : float
        Dropout rate on features, default: ``0``.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.

    Examples
    --------
    >>> import dgl
    >>> import torch
    >>> from cugraph_dgl.nn import SAGEConv
    ...
    >>> device = 'cuda'
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3])).to(device)
    >>> g = dgl.add_self_loop(g)
    >>> feat = torch.ones(6, 10).to(device)
    >>> conv = SAGEConv(10, 2, 'mean').to(device)
    >>> res = conv(g, feat)
    >>> res
    tensor([[-1.1690,  0.1952],
            [-1.1690,  0.1952],
            [-1.1690,  0.1952],
            [-1.1690,  0.1952],
            [-1.1690,  0.1952],
            [-1.1690,  0.1952]], device='cuda:0', grad_fn=<AddmmBackward0>)
    """
    valid_aggr_types = {"mean", "sum", "min", "max", "pool", "gcn"}

    def __init__(
        self,
        in_feats: Union[int, tuple[int, int]],
        out_feats: int,
        aggregator_type: str = "mean",
        feat_drop: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        if aggregator_type not in self.valid_aggr_types:
            raise ValueError(
                f"Invalid aggregator_type. Must be one of {self.valid_aggr_types}. "
                f"But got '{aggregator_type}' instead."
            )

        self.aggregator_type = aggregator_type
        self._aggr = aggregator_type
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.in_feats_src, self.in_feats_dst = dgl.utils.expand_as_pair(in_feats)
        self.feat_drop = nn.Dropout(feat_drop)

        if self.aggregator_type == "gcn":
            self._aggr = "mean"
            self.lin = nn.Linear(self.in_feats_src, out_feats, bias=bias)
        else:
            self.lin = nn.Linear(
                self.in_feats_src + self.in_feats_dst, out_feats, bias=bias
            )

        if self.aggregator_type == "pool":
            self._aggr = "max"
            self.pre_lin = nn.Linear(self.in_feats_src, self.in_feats_src)
        else:
            self.register_parameter("pre_lin", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Reinitialize learnable parameters."""
        self.lin.reset_parameters()
        if self.pre_lin is not None:
            self.pre_lin.reset_parameters()

    def forward(
        self,
        g: Union[SparseGraph, dgl.DGLHeteroGraph],
        feat: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        max_in_degree: Optional[int] = None,
    ) -> torch.Tensor:
        r"""Forward computation.

        Parameters
        ----------
        g : DGLGraph or SparseGraph
            The graph.
        feat : torch.Tensor or tuple
            Node features. Shape: :math:`(|V|, D_{in})`.
        max_in_degree : int
            Maximum in-degree of destination nodes. When :attr:`g` is generated
            from a neighbor sampler, the value should be set to the corresponding
            :attr:`fanout`. This option is used to invoke the MFG-variant of
            cugraph-ops kernel.

        Returns
        -------
        torch.Tensor
            Output node features. Shape: :math:`(|V|, D_{out})`.
        """
        feat_bipartite = isinstance(feat, (list, tuple))
        graph_bipartite = feat_bipartite or self.aggregator_type == "pool"

        _graph = self.get_cugraph_ops_CSC(
            g, is_bipartite=graph_bipartite, max_in_degree=max_in_degree
        )

        if feat_bipartite:
            feat = (self.feat_drop(feat[0]), self.feat_drop(feat[1]))
        else:
            feat = self.feat_drop(feat)

        if self.aggregator_type == "pool":
            if feat_bipartite:
                feat = (self.pre_lin(feat[0]).relu(), feat[1])
            else:
                feat = (self.pre_lin(feat).relu(), feat[: g.num_dst_nodes()])
            # force ctx.needs_input_grad=True in cugraph-ops autograd function
            feat[0].requires_grad_()
            feat[1].requires_grad_()

        out = ops_torch.operators.agg_concat_n2n(feat, _graph, self._aggr)[
            : g.num_dst_nodes()
        ]

        if self.aggregator_type == "gcn":
            out = out[:, : self.in_feats_src]

        out = self.lin(out)

        return out
