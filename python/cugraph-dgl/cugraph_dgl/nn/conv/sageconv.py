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
from __future__ import annotations
from typing import Optional, Union

from cugraph_dgl.nn.conv.base import BaseConv, SparseGraph
from cugraph.utilities.utils import import_optional

dgl = import_optional("dgl")
torch = import_optional("torch")
nn = import_optional("torch.nn")
ops_torch = import_optional("pylibcugraphops.pytorch")


class SAGEConv(BaseConv):
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
    MAX_IN_DEGREE_MFG = 500

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        aggregator_type: str = "mean",
        feat_drop: float = 0.0,
        bias: bool = True,
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

        self.linear = nn.Linear(2 * in_feats, out_feats, bias=bias)

    def reset_parameters(self):
        r"""Reinitialize learnable parameters."""
        self.linear.reset_parameters()

    def forward(
        self,
        g: Union[SparseGraph, dgl.DGLHeteroGraph],
        feat: torch.Tensor,
        max_in_degree: Optional[int] = None,
    ) -> torch.Tensor:
        r"""Forward computation.

        Parameters
        ----------
        g : DGLGraph or SparseGraph
            The graph.
        feat : torch.Tensor
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
        if max_in_degree is None:
            max_in_degree = -1

        if isinstance(g, SparseGraph):
            assert "csc" in g.formats()
            offsets, indices = g.csc()
            _graph = ops_torch.CSC(
                offsets=offsets,
                indices=indices,
                num_src_nodes=g.num_src_nodes(),
                dst_max_in_degree=max_in_degree,
            )
        elif isinstance(g, dgl.DGLHeteroGraph):
            offsets, indices, _ = g.adj_tensors("csc")
            _graph = ops_torch.CSC(
                offsets=offsets,
                indices=indices,
                num_src_nodes=g.num_src_nodes(),
                dst_max_in_degree=max_in_degree,
            )
        else:
            raise TypeError(
                f"The graph has to be either a 'SparseGraph' or "
                f"'dgl.DGLHeteroGraph', but got '{type(g)}'."
            )

        feat = self.feat_drop(feat)
        h = ops_torch.operators.agg_concat_n2n(feat, _graph, self.aggr)[
            : g.num_dst_nodes()
        ]
        h = self.linear(h)

        return h
