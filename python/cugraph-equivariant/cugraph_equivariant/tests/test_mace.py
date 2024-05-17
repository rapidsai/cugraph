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
from cugraph_equivariant.nn.mace import InteractionBlock, FusedSymmetricContraction
from utils import get_random_graph
import torch
from e3nn import o3

device = torch.device("cuda")


@pytest.mark.parametrize("irreps_in", ["32x0e", "64x0e+64x1o"])
def test_interaction_block_equivariance(irreps_in):
    num_nodes = (100, 100)
    num_batches = 1000
    graph = get_random_graph(*num_nodes, num_batches, device=device)

    num_elements = 2
    num_basis = 8
    avg_number_neighbors = 8

    in_irreps = o3.Irreps(irreps_in)
    sh_irreps = o3.Irreps.spherical_harmonics(2)
    num_features = 32
    target_irreps = (sh_irreps * num_features).sort()[0].simplify()

    node_feats = torch.randn(num_nodes[0], in_irreps.dim, device=device)
    node_attrs = torch.randn(num_nodes[0], num_elements, device=device)
    edge_feats = torch.randn(num_batches, num_basis, device=device)
    edge_attrs = torch.randn(num_batches, sh_irreps.dim, device=device)

    conv = InteractionBlock(
        in_irreps=in_irreps,
        sh_irreps=sh_irreps,
        target_irreps=target_irreps,
        num_elements=num_elements,
        num_bessel_basis=num_basis,
        avg_num_neighbors=avg_number_neighbors,
        e3nn_compat_mode=True,
    ).to(device=device)

    rot = o3.rand_matrix()
    D_in = conv.in_irreps.D_from_matrix(rot).to(device)
    D_sh = conv.sh_irreps.D_from_matrix(rot).to(device)
    D_out = conv.target_irreps.D_from_matrix(rot).to(device)

    out_before = conv(
        node_feats @ D_in.T, node_attrs, edge_feats, edge_attrs @ D_sh.T, graph
    )

    out_after = conv(node_feats, node_attrs, edge_feats, edge_attrs, graph) @ D_out.T

    assert torch.allclose(out_before, out_after, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("multiplicity", [16, 32, 128])
def test_symmetric_contraction(multiplicity):
    mul = multiplicity
    irreps_in = o3.Irreps(f"{mul}x0e + {mul}x1o + {mul}x2e")
    irreps_out = o3.Irreps(f"{mul}x0e + {mul}x1o")
    num_element = 2
    f = FusedSymmetricContraction(
        irreps_in,
        irreps_out,
        correlation=3,
        num_elements=num_element,
        e3nn_compat_mode=True,
    ).to(device)

    batch_size = 128
    feats = torch.randn(batch_size, mul, int(irreps_in.dim / mul), device=device)

    one_hots = torch.nn.functional.one_hot(
        torch.arange(0, batch_size) % num_element
    ).to(device=device, dtype=feats.dtype)

    out = f(feats, one_hots)
    assert out.size(1) == irreps_out.dim
