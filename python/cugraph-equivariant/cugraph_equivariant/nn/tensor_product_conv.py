# Copyright (c) 2024, NVIDIA CORPORATION.
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

from typing import Optional, Sequence, Union

import torch
from torch import nn
from e3nn import o3
from e3nn.nn import BatchNorm

from cugraph_equivariant.utils import scatter_reduce

from pylibcugraphops.pytorch.operators import FusedFullyConnectedTensorProduct


class FullyConnectedTensorProductConv(nn.Module):
    r"""Message passing layer for tensor products in DiffDock-like architectures.
    The left operand of tensor product is the spherical harmonic representation
    of edge vector; the right operand consists of node features in irreps.

    .. math::
        \sum_{b \in \mathcal{N}_a} Y\left(\hat{r}_{a b}\right)
        \otimes_{\psi_{a b}} \mathbf{h}_b

    where the path weights :math:`\psi_{a b}` can be constructed from edge
    embeddings and scalar features using an MLP:

    .. math::
        \psi_{a b} = \operatorname{MLP}
        \left(e_{a b}, \mathbf{h}_a^0, \mathbf{h}_b^0\right)

    Users have the option to either directly input the weights or provide the
    MLP parameters and scalar features from edges and nodes.

    Parameters
    ----------
    in_irreps : e3nn.o3.Irreps
        Irreps for the input node features.

    sh_irreps : e3nn.o3.Irreps
        Irreps for the spherical harmonic representations of edge vectors.

    out_irreps : e3nn.o3.Irreps
        Irreps for the output.

    batch_norm : bool, optional (default=True)
        If true, batch normalization is applied.

    mlp_channels : sequence of ints, optional (default=None)
        A sequence of integers defining number of neurons in each layer in MLP
        before the output layer. If `None`, no MLP will be added. The input layer
        contains edge embeddings and node scalar features.

    mlp_activation : nn.Module or sequence of nn.Module, optional (default=nn.GELU())
        A sequence of functions to be applied in between linear layers in MLP,
        e.g., `nn.Sequential(nn.ReLU(), nn.Dropout(0.4))`.

    e3nn_compat_mode: bool, optional (default=False)
        cugraph-ops and e3nn use different memory layout for Irreps-tensors.
        The last (fastest moving) dimension is num_channels for cugraph-ops and
        ir.dim for e3nn. When enabled, the input and output of this layer will
        follow e3nn's memory layout.

    Examples
    --------
    >>> # Case 1: MLP with the input layer having 6 channels and 2 hidden layers
    >>> #         having 16 channels. edge_emb.size(1) must match the size of
    >>> #         the input layer: 6
    >>>
    >>> conv1 = FullyConnectedTensorProductConv(in_irreps, sh_irreps, out_irreps,
    >>>     mlp_channels=[6, 16, 16], mlp_activation=nn.ReLU()).cuda()
    >>> out = conv1(src_features, edge_sh, edge_emb, graph)
    >>>
    >>> # Case 2: Same as case 1 but with the scalar features from edges, sources
    >>> #         and destinations passed in separately.
    >>>
    >>> conv2 = FullyConnectedTensorProductConv(in_irreps, sh_irreps, out_irreps,
    >>>     mlp_channels=[6, 16, 16], mlp_activation=nn.ReLU()).cuda()
    >>> out = conv3(src_features, edge_sh, edge_scalars, graph,
    >>>     src_scalars=src_scalars, dst_scalars=dst_scalars)
    >>>
    >>> # Case 3: No MLP, edge_emb will be directly used as the tensor product weights
    >>>
    >>> conv3 = FullyConnectedTensorProductConv(in_irreps, sh_irreps, out_irreps,
    >>>     mlp_channels=None).cuda()
    >>> out = conv2(src_features, edge_sh, edge_emb, graph)

    """

    def __init__(
        self,
        in_irreps: o3.Irreps,
        sh_irreps: o3.Irreps,
        out_irreps: o3.Irreps,
        batch_norm: bool = True,
        mlp_channels: Optional[Sequence[int]] = None,
        mlp_activation: Union[nn.Module, Sequence[nn.Module]] = nn.GELU(),
        e3nn_compat_mode: bool = False,
    ):
        super().__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.e3nn_compat_mode = e3nn_compat_mode

        self.tp = FusedFullyConnectedTensorProduct(
            in_irreps, sh_irreps, out_irreps, e3nn_compat_mode=e3nn_compat_mode
        )

        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

        if mlp_activation is None:
            mlp_activation = []
        elif hasattr(mlp_activation, "__len__") and hasattr(
            mlp_activation, "__getitem__"
        ):
            mlp_activation = list(mlp_activation)
        else:
            mlp_activation = [mlp_activation]

        if mlp_channels is not None:
            dims = list(mlp_channels) + [self.tp.weight_numel]
            mlp = []
            for i in range(len(dims) - 1):
                mlp.append(nn.Linear(dims[i], dims[i + 1]))
                if i != len(dims) - 2:
                    mlp.extend(mlp_activation)
            self.mlp = nn.Sequential(*mlp)
        else:
            self.mlp = None

    def forward(
        self,
        src_features: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_emb: torch.Tensor,
        graph: tuple[torch.Tensor, tuple[int, int]],
        src_scalars: Optional[torch.Tensor] = None,
        dst_scalars: Optional[torch.Tensor] = None,
        reduce: str = "mean",
        edge_envelope: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        src_features : torch.Tensor
            Source node features.
            Shape: (num_src_nodes, in_irreps.dim)

        edge_sh : torch.Tensor
            The spherical harmonic representations of the edge vectors.
            Shape: (num_edges, sh_irreps.dim)

        edge_emb: torch.Tensor
            Edge embeddings that are fed into MLPs to generate tensor product weights.
            Shape: (num_edges, dim), where `dim` should be:
            - `tp.weight_numel` when the layer does not contain MLPs.
            - num_edge_scalars, with the sum of num_[edge/src/dst]_scalars being
              mlp_channels[0]

        graph : tuple
            A tuple that stores the graph information, with the first element being
            the adjacency matrix in COO, and the second element being its shape:
            (num_src_nodes, num_dst_nodes).

        src_scalars: torch.Tensor, optional
            Scalar features of source nodes.
            Shape: (num_src_nodes, num_src_scalars)

        dst_scalars: torch.Tensor, optional
            Scalar features of destination nodes.
            Shape: (num_dst_nodes, num_dst_scalars)

        reduce : str, optional (default="mean")
            Reduction operator. Choose between "mean" and "sum".

        edge_envelope: torch.Tensor, optional
            Typically used as attenuation factors to fade out messages coming
            from nodes close to the cutoff distance used to create the graph.
            This is important to make the model smooth to the changes in node's
            coordinates.
            Shape: (num_edges,)

        Returns
        -------
        torch.Tensor
            Output node features.
            Shape: (num_dst_nodes, out_irreps.dim)
        """
        edge_emb_size = edge_emb.size(-1)
        src_scalars_size = 0 if src_scalars is None else src_scalars.size(-1)
        dst_scalars_size = 0 if dst_scalars is None else dst_scalars.size(-1)

        if self.mlp is None:
            if self.tp.weight_numel != edge_emb_size:
                raise RuntimeError(
                    f"When MLP is not present, edge_emb's last dimension must "
                    f"equal tp.weight_numel (but got {edge_emb_size} and "
                    f"{self.tp.weight_numel})"
                )
        else:
            total_size = edge_emb_size + src_scalars_size + dst_scalars_size
            if self.mlp[0].in_features != total_size:
                raise RuntimeError(
                    f"The size of MLP's input layer ({self.mlp[0].in_features}) "
                    f"does not match the total number of scalar features from "
                    f"edge_emb, src_scalars and dst_scalars ({total_size})"
                )

        if reduce not in ["mean", "sum"]:
            raise RuntimeError(
                f"reduce argument must be either 'mean' or 'sum', got {reduce}."
            )

        (src, dst), (num_src_nodes, num_dst_nodes) = graph

        if self.mlp is not None:
            if src_scalars is None and dst_scalars is None:
                tp_weights = self.mlp(edge_emb)
            else:
                w_edge, w_src, w_dst = torch.split(
                    self.mlp[0].weight,
                    (edge_emb_size, src_scalars_size, dst_scalars_size),
                    dim=-1,
                )
                tp_weights = edge_emb @ w_edge.T + self.mlp[0].bias

                if src_scalars is not None:
                    tp_weights += (src_scalars @ w_src.T)[src]

                if dst_scalars is not None:
                    tp_weights += (dst_scalars @ w_dst.T)[dst]

                tp_weights = self.mlp[1:](tp_weights)
        else:
            tp_weights = edge_emb

        out = self.tp(src_features[src], edge_sh, tp_weights)

        if edge_envelope is not None:
            out = out * edge_envelope.view(-1, 1)

        out = scatter_reduce(out, dst, dim=0, dim_size=num_dst_nodes, reduce=reduce)

        if self.batch_norm:
            out = self.batch_norm(out)

        return out
