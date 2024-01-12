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

    mlp_fast_first_layer : bool, optional (default=False)
        When enabled, for the first layer of MLP, the module performs GEMMs on
        individual components (i.e., edge embeddings, scalar features of source
        and destinations nodes) of the weights before indexing and concatenation,
        leading to a lower complexity in most use cases. This option requires
        users to explicitly pass in `src_scalars` and `dst_scalars` in
        `forward()` call.
    """

    def __init__(
        self,
        in_irreps: o3.Irreps,
        sh_irreps: o3.Irreps,
        out_irreps: o3.Irreps,
        batch_norm: bool = True,
        mlp_channels: Optional[Sequence[int]] = None,
        mlp_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
        mlp_fast_first_layer: bool = False,
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

        if mlp_fast_first_layer:
            assert mlp_channels is not None
            assert mlp_channels[0] % 3 == 0
            self.num_scalars = int(mlp_channels[0] / 3)
        else:
            self.num_scalars = None
        self.mlp_fast_first_layer = mlp_fast_first_layer

        if mlp_channels is not None:
            dims = list(mlp_channels) + [self.tp.weight_numel]
            mlp = []
            for i in range(len(dims) - 1):
                mlp.append(nn.Linear(dims[i], dims[i + 1]))
                if mlp_activation is not None and i != len(dims) - 2:
                    mlp.append(mlp_activation())
            self.mlp = nn.Sequential(*mlp)
        else:
            self.mlp = None

    def forward(
        self,
        src_features: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_emb: torch.Tensor,
        graph: tuple[torch.Tensor, tuple[int, int]],  # COO, (n_src, n_dst)
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
            Shape: (num_edges, dim), where `dim` should equal to:
            - `tp.weight_numel` when the layer does not contain MLPs.
            - `num_scalars` when `mlp_fast_first_layer` enabled.
            - `mlp_channels[0]` otherwise.

        graph : Any
            The graph.

        src_scalars: torch.Tensor, optional
            Scalar features of source nodes.
            Shape: (num_src_nodes, num_scalars)

        dst_scalars: torch.Tensor, optional
            Scalar features of destination nodes.
            Shape: (num_dst_nodes, num_scalars)

        reduce : str, optional (default="mean")
            Reduction operator. Choose between "mean" and "sum".

        edge_envelope: torch.Tensor, optional
            Edge envelope.
            Shape: (num_edges,)

        Returns
        -------
        torch.Tensor
            Output node features.
            Shape: (num_dst_nodes, out_irreps.dim)
        """
        if self.mlp is None:
            assert self.tp.weight_numel == edge_emb.size(-1)
        else:
            if self.mlp_fast_first_layer:
                assert edge_emb.size(-1) == self.num_scalars
                assert src_scalars.size(-1) == self.num_scalars
                assert dst_scalars.size(-1) == self.num_scalars
            else:
                assert self.mlp[0].in_features == edge_emb.size(-1)

        if reduce not in ["mean", "sum"]:
            raise ValueError(
                f"reduce argument must be either 'mean' or 'sum', got {reduce}."
            )

        (src, dst), (num_src_nodes, num_dst_nodes) = graph

        if self.mlp is not None:
            if self.mlp_fast_first_layer:
                w_edge, w_src, w_dst = torch.chunk(self.mlp[0].weight, chunks=3, dim=-1)
                tp_weights = (
                    edge_emb @ w_edge.T
                    + (src_scalars @ w_src.T)[src]
                    + (dst_scalars @ w_dst.T)[dst]
                    + self.mlp[0].bias
                )
                tp_weights = self.mlp[1:](tp_weights)
            else:
                tp_weights = self.mlp(edge_emb)
        else:
            tp_weights = edge_emb

        out = self.tp(src_features[src], edge_sh, tp_weights)

        if edge_envelope is not None:
            out = out * edge_envelope.view(-1, 1)

        out = scatter_reduce(out, dst, dim=0, dim_size=num_dst_nodes, reduce=reduce)

        if self.batch_norm:
            out = self.batch_norm(out)

        return out
