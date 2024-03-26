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

import itertools
import pytest
import torch
from torch import nn
from e3nn import o3
from cugraph_equivariant.nn import FullyConnectedTensorProductConv
from typing import Optional

device = torch.device("cuda:0")

@pytest.fixture
def example_scatter_data():
    src_feat = torch.Tensor([3, 1, 0, 1, 1, 2])
    dst_indices = torch.Tensor([0, 1, 2, 2, 3, 1])

    results = {
        "sum": torch.Tensor([3.0, 3.0, 1.0, 1.0]),
        "mean": torch.Tensor([3.0, 1.5, 0.5, 1.0]),
        "prod": torch.Tensor([3.0, 2.0, 0.0, 1.0]),
        "amax": torch.Tensor([3.0, 2.0, 1.0, 1.0]),
        "amin": torch.Tensor([3.0, 1.0, 0.0, 1.0]),
    }

    return src_feat, dst_indices, results

@pytest.fixture
def empty_scatter_data():
    src_feat = torch.empty((0, 41))
    dst_indices = torch.empty((0,))

    return src_feat, dst_indices

e3nn_compat_mode = [True, False]
batch_norm = [True, False]
MLP=[
    [(30, 8, 8), nn.Sequential(nn.Dropout(0.3), nn.ReLU()), (15, 15, 0)],
    [(7,), nn.GELU(), (2, 3, 2)],
    [None, None, None],
]

BatchCompatMLP=itertools.product(batch_norm, e3nn_compat_mode, MLP)

@pytest.fixture(scope="module", params=BatchCompatMLP)
def create_tp_conv(request):
    (batch_norm, e3nn_compat_mode, (mlp_channels, mlp_activation, scalar_sizes)) = request.param
    # TODO: parameterize
    # mlp_channels=[(30, 8, 8), nn.Sequential(nn.Dropout(0.3), nn.ReLU()), (15, 15, 0)]
    # mlp_activation=[(7,), nn.GELU(), (2, 3, 2)]
    # scalar_sizes=[None, None, None]
    
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
    ).to(device)
    tp_conv.eval()
    return tp_conv, request.param
    
@pytest.fixture(scope="module")
def create_tp_conv_and_data(create_tp_conv):
    tp_conv, param = create_tp_conv
    (batch_norm, e3nn_compat_mode, (mlp_channels, mlp_activation, scalar_sizes)) = param
    num_src_nodes, num_dst_nodes = 9, 7
    num_edges = 40
    src = torch.randint(num_src_nodes, (num_edges,), device=device)
    dst = torch.randint(num_dst_nodes, (num_edges,), device=device)
    edge_index = torch.vstack((src, dst))

    src_pos = torch.randn(num_src_nodes, 3, device=device)
    dst_pos = torch.randn(num_dst_nodes, 3, device=device)
    edge_vec = dst_pos[dst] - src_pos[src]
    edge_sh = o3.spherical_harmonics(
        tp_conv.sh_irreps, edge_vec, normalize=True, normalization="component"
    ).to(device)
    src_features = torch.randn(num_src_nodes, tp_conv.in_irreps.dim, device=device)
    rot = o3.rand_matrix()
    D_in = tp_conv.in_irreps.D_from_matrix(rot).to(device)
    D_sh = tp_conv.sh_irreps.D_from_matrix(rot).to(device)
    D_out = tp_conv.out_irreps.D_from_matrix(rot).to(device)

    if mlp_channels is None:
        edge_emb = torch.randn(num_edges, tp_conv.tp.weight_numel, device=device)
        src_scalars = dst_scalars = None
    else:
        if scalar_sizes:
            edge_emb = torch.randn(num_edges, scalar_sizes[0], device=device)
            src_scalars = (
                None
                if scalar_sizes[1] == 0
                else torch.randn(num_src_nodes, scalar_sizes[1], device=device)
            )
            dst_scalars = (
                None
                if scalar_sizes[2] == 0
                else torch.randn(num_dst_nodes, scalar_sizes[2], device=device)
            )
        else:
            edge_emb = torch.randn(num_edges, tp_conv.mlp[0].in_features, device=device)
            src_scalars = dst_scalars = None

    return (tp_conv, (src_features, edge_sh, edge_emb, edge_index, torch.empty((num_dst_nodes, 0)),
                      src_scalars, dst_scalars), (D_in, D_sh, D_out))
    
