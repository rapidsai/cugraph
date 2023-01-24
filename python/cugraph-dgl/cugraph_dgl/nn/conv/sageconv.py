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
"""Torch Module for GraphSAGE layer using the aggregation primitives in
cugraph-ops"""
# pylint: disable=no-member, arguments-differ, invalid-name, too-many-arguments

from cugraph.utilities.utils import import_optional

torch = import_optional("torch")
nn = import_optional("torch.nn")
ops = import_optional("pylibcugraphops")
ops_autograd = import_optional("pylibcugraphops.torch.autograd")


class SAGEConv(nn.Module):
    r"""An accelerated GraphSAGE layer from `Inductive Representation Learning
    on Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__ that leverages the
    highly-optimized aggregation primitives in cugraph-ops.

    See :class:`dgl.nn.pytorch.conv.SAGEConv` for mathematical model.

    This module depends on :code:`pylibcugraphops` package, which can be
    installed via :code:`conda install -c nvidia pylibcugraphops>=23.02`.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    aggregator_type : str
        Aggregator type to use (``mean``, ``sum``, ``min``, ``max``).
    feat_drop : float
        Dropout rate on features, default: ``0``.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.

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

    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type="mean",
        feat_drop=0.0,
        bias=True,
        norm=None,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        valid_aggr_types = {"max", "min", "mean", "sum"}
        if aggregator_type not in valid_aggr_types:
            raise ValueError(
                f"Invalid aggregator_type. Must be one of {valid_aggr_types}. "
                f"But got '{aggregator_type}' instead."
            )
        self.aggr = aggregator_type
        self.feat_drop = nn.Dropout(feat_drop)
        self.norm = norm

        self.linear = nn.Linear(2 * in_feats, out_feats, bias=bias)

    def reset_parameters(self):
        r"""Reinitialize learnable parameters."""
        self.linear.reset_parameters()

    def forward(self, g, feat, max_in_degree=None):
        r"""Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            Node features. Shape: :math:`(|V|, D_{in})`.
        max_in_degree : int
            Maximum in-degree of destination nodes. It is only effective when
            :attr:`g` is a :class:`DGLBlock`, i.e., bipartite graph. When
            :attr:`g` is generated from a neighbor sampler, the value should be
            set to the corresponding :attr:`fanout`. If not given,
            :attr:`max_in_degree` will be calculated on-the-fly.

        Returns
        -------
        torch.Tensor
            Output node features. Shape: :math:`(|V|, D_{out})`.
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

        feat = self.feat_drop(feat)
        h = ops_autograd.agg_concat_n2n(feat, _graph, self.aggr)
        h = self.linear(h)

        if self.norm is not None:
            h = self.norm(h)

        return h
