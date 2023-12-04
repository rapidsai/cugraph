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

from typing import Optional, Callable, Sequence

import torch
from torch import nn
from e3nn import o3
from e3nn.nn import BatchNorm

from cugraph_equivariant.utils import scatter_reduce

try:
    from pylibcugraphops.equivariant import TensorProduct

    HAS_TP_LIB = True
except ImportError:
    HAS_TP_LIB = False


class FullyConnectedTensorProductConv(nn.Module):
    r"""Message passing layer for tensor products in DiffDock-like architectures.
    The left operand of tensor product is the spherical harmonic representation
    of edge vector; the right operand consists of node features in irreps.

    .. math::
        \sum_{b \in \mathcal{N}_a} Y\left(\hat{r}_{a b}\right)
        \otimes_{\psi_{a b}} \mathbf{h}_b

    where the path weights :math:`\psi_{a b}` are from user input to the forward()
    function. For example, they can be constructed from edge embeddings and
    scalar features:

    .. math::
        \psi_{a b} = \operatorname{MLP}
        \left(e_{a b}, \mathbf{h}_a^0, \mathbf{h}_b^0\right)

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
        before the output layer. If `None`, no MLP will be added.

    mlp_activation : callable, optional (default=torch.nn.GELU)
        Activation function in MLP.
    """

    def __init__(
        self,
        in_irreps: o3.Irreps,
        sh_irreps: o3.Irreps,
        out_irreps: o3.Irreps,
        batch_norm: bool = True,
        mlp_channels: Optional[Sequence[int]] = None,
        mlp_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
    ):
        super().__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps

        if HAS_TP_LIB:
            self.tp = TensorProduct(str(in_irreps), str(sh_irreps), str(out_irreps))
        else:
            self.tp = o3.FullyConnectedTensorProduct(
                in_irreps, sh_irreps, out_irreps, shared_weights=False
            )

        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

        if mlp_channels is not None:
            dims = list(mlp_channels) + [self.tp.weight_numel]
            mlp = []
            for i in range(len(dims) - 1):
                mlp.append(nn.Linear(dims[i], dims[i + 1]))
                if mlp_activation is not None and i != len(dims) - 2:
                    mlp.append(mlp_activation)
            self.mlp = nn.Sequential(mlp)
        else:
            self.mlp = None

    def forward(
        self,
        src_features: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_scalars: torch.Tensor,  # (n_edge, n_edge_scalars)
        graph: tuple[torch.Tensor, tuple[int, int]],  # COO, (n_src, n_dst)
        src_scalars: Optional[torch.Tensor] = None,  # (n_src, n_src_scalars)
        dst_scalars: Optional[torch.Tensor] = None,  # (n_dst, n_dst_scalars)
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

        edge_scalars: torch.Tensor
            Edge embeddings.
            Shape: (num_edges, self.tp.weight_numel) when the layer does not
            contain MLP; (num_edges, self.mlp[0].in_features) when it does.

        graph : Any
            The graph.

        src_scalars:
        dst_scalars:

        reduce : str, optional (default="mean")
            Reduction operator. Choose between "mean" and "sum".

        Returns
        -------
        torch.Tensor
            Output node features.
            Shape: (num_dst_nodes, out_irreps.dim)
        """
        if self.mlp is None:
            assert self.tp.weight_numel == edge_scalars.size(-1)
        else:
            assert self.mlp[0].in_features == edge_scalars.size(-1)

        if reduce not in ["mean", "sum"]:
            raise ValueError(
                f"reduce argument must be either 'mean' or 'sum', got {reduce}."
            )

        (src, dst), (num_src_nodes, num_dst_nodes) = graph

        if self.mlp is not None:
            tp_weights = self.mlp(edge_scalars)
        else:
            tp_weights = edge_scalars

        out_tp = self.tp(src_features[src], edge_sh, tp_weights)
        out = scatter_reduce(out_tp, dst, dim=0, dim_size=num_dst_nodes, reduce=reduce)

        if self.batch_norm:
            out = self.batch_norm(out)

        return out
