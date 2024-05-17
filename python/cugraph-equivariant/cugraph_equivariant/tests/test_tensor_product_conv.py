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

import pytest
from utils import get_random_graph
import torch
from torch import nn
from e3nn import o3
from cugraph_equivariant.nn import FullyConnectedTensorProductConv

device = torch.device("cuda")


@pytest.mark.parametrize("e3nn_compat_mode", [True, False])
@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize(
    "mlp_channels, mlp_activation, scalar_sizes",
    [
        [(30, 8, 8), nn.Sequential(nn.Dropout(0.3), nn.ReLU()), (15, 15, 0)],
        [(7,), nn.GELU(), (2, 3, 2)],
        [None, None, None],
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_tensor_product_conv_equivariance(
    mlp_channels, mlp_activation, scalar_sizes, batch_norm, e3nn_compat_mode, dtype
):
    torch.manual_seed(12345)

    num_src_nodes, num_dst_nodes = 9, 7
    num_batches = 100
    graph = get_random_graph(num_src_nodes, num_dst_nodes, num_batches, device=device)

    in_irreps = o3.Irreps("10x0e + 10x1e")
    out_irreps = o3.Irreps("20x0e + 10x1e")
    sh_irreps = o3.Irreps.spherical_harmonics(lmax=2)

    tp_conv = FullyConnectedTensorProductConv(
        in_irreps=in_irreps,
        sh_irreps=sh_irreps,
        out_irreps=out_irreps,
        mlp_channels=mlp_channels,
        mlp_activation=mlp_activation,
        batch_norm=batch_norm,
        e3nn_compat_mode=e3nn_compat_mode,
    ).to(device=device, dtype=dtype)

    edge_sh = torch.randn(num_batches, sh_irreps.dim, device=device, dtype=dtype)
    src_features = torch.randn(num_src_nodes, in_irreps.dim, device=device, dtype=dtype)

    rot = o3.rand_matrix()
    D_in = tp_conv.in_irreps.D_from_matrix(rot).to(device=device, dtype=dtype)
    D_sh = tp_conv.sh_irreps.D_from_matrix(rot).to(device=device, dtype=dtype)
    D_out = tp_conv.out_irreps.D_from_matrix(rot).to(device=device, dtype=dtype)

    if mlp_channels is None:
        edge_emb = torch.randn(
            num_batches, tp_conv.tp.weight_numel, device=device, dtype=dtype
        )
        src_scalars = dst_scalars = None
    else:
        if scalar_sizes:
            edge_emb = torch.randn(
                num_batches, scalar_sizes[0], device=device, dtype=dtype
            )
            src_scalars = (
                None
                if scalar_sizes[1] == 0
                else torch.randn(
                    num_src_nodes, scalar_sizes[1], device=device, dtype=dtype
                )
            )
            dst_scalars = (
                None
                if scalar_sizes[2] == 0
                else torch.randn(
                    num_dst_nodes, scalar_sizes[2], device=device, dtype=dtype
                )
            )
        else:
            edge_emb = torch.randn(
                num_batches, tp_conv.mlp[0].in_features, device=device, dtype=dtype
            )
            src_scalars = dst_scalars = None

    # rotate before
    torch.manual_seed(12345)
    out_before = tp_conv(
        src_features=src_features @ D_in.T,
        edge_sh=edge_sh @ D_sh.T,
        edge_emb=edge_emb,
        graph=graph,
        src_scalars=src_scalars,
        dst_scalars=dst_scalars,
    )

    # rotate after
    torch.manual_seed(12345)
    out_after = (
        tp_conv(
            src_features=src_features,
            edge_sh=edge_sh,
            edge_emb=edge_emb,
            graph=graph,
            src_scalars=src_scalars,
            dst_scalars=dst_scalars,
        )
        @ D_out.T
    )

    atol = 1e-3 if dtype == torch.float32 else 1e-1
    if e3nn_compat_mode:
        assert torch.allclose(out_before, out_after, rtol=1e-4, atol=atol)
