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

import torch
from e3nn import o3
from cugraph_equivariant.nn import FullyConnectedTensorProductConv


def test_tensor_product_conv_equivariance():
    torch.manual_seed(12345)

    in_irreps = o3.Irreps("10x0e + 10x1e")
    out_irreps = o3.Irreps("20x0e + 10x1e")
    sh_irreps = o3.Irreps.spherical_harmonics(lmax=2)

    tp_conv = FullyConnectedTensorProductConv(
        in_irreps=in_irreps, sh_irreps=sh_irreps, out_irreps=out_irreps
    )

    num_src_nodes, num_dst_nodes = 9, 7
    num_edges = 40
    src = torch.randint(num_src_nodes, (num_edges,))
    dst = torch.randint(num_dst_nodes, (num_edges,))
    edge_index = torch.vstack((src, dst))

    src_pos = torch.randn(num_src_nodes, 3)
    dst_pos = torch.randn(num_dst_nodes, 3)
    edge_vec = dst_pos[dst] - src_pos[src]
    edge_sh = o3.spherical_harmonics(
        tp_conv.sh_irreps, edge_vec, normalize=True, normalization="component"
    )
    src_features = torch.randn(num_src_nodes, in_irreps.dim)

    weights_tp = torch.randn(num_edges, tp_conv.tp.weight_numel)

    rot = o3.rand_matrix()
    D_in = tp_conv.in_irreps.D_from_matrix(rot)
    D_sh = tp_conv.sh_irreps.D_from_matrix(rot)
    D_out = tp_conv.out_irreps.D_from_matrix(rot)

    # rotate before
    out_before = tp_conv(
        src_features=src_features @ D_in,
        edge_sh=edge_sh @ D_sh,
        edge_emb=weights_tp,
        graph=(edge_index, (num_src_nodes, num_dst_nodes)),
    )

    # rotate after
    out_after = (
        tp_conv(
            src_features=src_features,
            edge_sh=edge_sh,
            edge_emb=weights_tp,
            graph=(edge_index, (num_src_nodes, num_dst_nodes)),
        )
        @ D_out
    )

    torch.allclose(out_before, out_after, rtol=1e-4, atol=1e-4)
