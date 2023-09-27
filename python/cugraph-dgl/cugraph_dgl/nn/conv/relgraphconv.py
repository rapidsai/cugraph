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

import math
from typing import Optional, Union

from cugraph_dgl.nn.conv.base import BaseConv, SparseGraph
from cugraph.utilities.utils import import_optional

dgl = import_optional("dgl")
torch = import_optional("torch")
nn = import_optional("torch.nn")
ops_torch = import_optional("pylibcugraphops.pytorch")


class RelGraphConv(BaseConv):
    r"""An accelerated relational graph convolution layer from `Modeling
    Relational Data with Graph Convolutional Networks
    <https://arxiv.org/abs/1703.06103>`__, with the sparse aggregation
    accelerated by cugraph-ops.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    num_rels : int
        Number of relations.
    regularizer : str, optional
        Which weight regularizer to use ("basis" or ``None``):
         - "basis" is for basis-decomposition.
         - ``None`` applies no regularization.
        Default: ``None``.
    num_bases : int, optional
        Number of bases. It comes into effect when a regularizer is applied.
        Default: ``None``.
    bias : bool, optional
        True if bias is added. Default: ``True``.
    self_loop : bool, optional
        True to include self loop message. Default: ``True``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``.
    apply_norm : bool, optional
        True to normalize aggregation output by the in-degree of the destination
        node per edge type, i.e. :math:`|\mathcal{N}^r_i|`. Default: ``True``.

    Examples
    --------
    >>> import dgl
    >>> import torch
    >>> from cugraph_dgl.nn import RelGraphConv
    ...
    >>> device = 'cuda'
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3])).to(device)
    >>> feat = torch.ones(6, 10).to(device)
    >>> conv = RelGraphConv(
    ...     10, 2, 3, regularizer='basis', num_bases=2).to(device)
    >>> etypes = torch.tensor([0,1,2,0,1,2]).to(device)
    >>> res = conv(g, feat, etypes)
    >>> res
    tensor([[-1.7774, -2.0184],
            [-1.4335, -2.3758],
            [-1.7774, -2.0184],
            [-0.4698, -3.0876],
            [-1.4335, -2.3758],
            [-1.4331, -2.3295]], device='cuda:0', grad_fn=<AddBackward0>)
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        num_rels: int,
        regularizer: Optional[str] = None,
        num_bases: Optional[int] = None,
        bias: bool = True,
        self_loop: bool = True,
        dropout: float = 0.0,
        apply_norm: bool = False,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_rels = num_rels
        self.apply_norm = apply_norm
        self.dropout = nn.Dropout(dropout)

        dim_self_loop = 1 if self_loop else 0
        self.self_loop = self_loop
        if regularizer is None:
            self.W = nn.Parameter(
                torch.Tensor(num_rels + dim_self_loop, in_feats, out_feats)
            )
            self.coeff = None
        elif regularizer == "basis":
            if num_bases is None:
                raise ValueError('Missing "num_bases" for basis regularization.')
            self.W = nn.Parameter(
                torch.Tensor(num_bases + dim_self_loop, in_feats, out_feats)
            )
            self.coeff = nn.Parameter(torch.Tensor(num_rels, num_bases))
            self.num_bases = num_bases
        else:
            raise ValueError(
                f"Supported regularizer options: 'basis' or None, but got "
                f"'{regularizer}'."
            )
        self.regularizer = regularizer

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Reinitialize learnable parameters."""
        bound = 1 / math.sqrt(self.in_feats)
        end = -1 if self.self_loop else None
        nn.init.uniform_(self.W[:end], -bound, bound)
        if self.regularizer == "basis":
            nn.init.xavier_uniform_(self.coeff, gain=nn.init.calculate_gain("relu"))
        if self.self_loop:
            nn.init.xavier_uniform_(self.W[-1], nn.init.calculate_gain("relu"))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        g: Union[SparseGraph, dgl.DGLHeteroGraph],
        feat: torch.Tensor,
        etypes: torch.Tensor,
        max_in_degree: Optional[int] = None,
    ) -> torch.Tensor:
        r"""Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        etypes : torch.Tensor
            A 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
            Note that cugraph-ops only accepts edge type tensors in int32,
            so any input of other integer types will be casted into int32,
            thus introducing some overhead. Pass in int32 tensors directly
            for best performance.
        max_in_degree : int
            Maximum in-degree of destination nodes. When :attr:`g` is generated
            from a neighbor sampler, the value should be set to the corresponding
            :attr:`fanout`. This option is used to invoke the MFG-variant of
            cugraph-ops kernel.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{out})`.
        """
        _graph = self.get_cugraph_ops_HeteroCSC(
            g,
            num_edge_types=self.num_rels,
            etypes=etypes,
            is_bipartite=False,
            max_in_degree=max_in_degree,
        )

        h = ops_torch.operators.agg_hg_basis_n2n_post(
            feat,
            self.coeff,
            _graph,
            concat_own=self.self_loop,
            norm_by_out_degree=self.apply_norm,
        )[: g.num_dst_nodes()]
        h = h @ self.W.view(-1, self.out_feats)
        if self.bias is not None:
            h = h + self.bias
        h = self.dropout(h)

        return h
