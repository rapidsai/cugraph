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

from cugraph_equivariant.nn.tensor_product_conv import MACE_InteractionBlock
from utils import get_random_edge_index
import torch
from e3nn import o3

device = torch.device("cuda")

sparse_size = (10, 10)
num_batches = 100
edge_index = get_random_edge_index(*sparse_size, num_batches, device=device)

num_elements = 2
num_basis = 8
avg_number_neighbors = 8

in_irreps = o3.Irreps("32x0e+32x1o")
sh_irreps = o3.Irreps.spherical_harmonics(2)
num_features = 32
target_irreps = (sh_irreps * num_features).sort()[0].simplify()

node_feats = torch.randn(sparse_size[0], in_irreps.dim, device=device)
node_attrs = torch.randn(sparse_size[0], num_elements, device=device)
edge_feats = torch.randn(num_batches, num_basis, device=device)
edge_attrs = torch.randn(num_batches, sh_irreps.dim, device=device)

conv = MACE_InteractionBlock(
    in_irreps=in_irreps,
    sh_irreps=sh_irreps,
    target_irreps=target_irreps,
    num_elements=num_elements,
    num_bessel_basis=num_basis,
    avg_num_neighbors=avg_number_neighbors,
    e3nn_compat_mode=True,
).to(device=device)

graph = ((edge_index[0], edge_index[1]), edge_index.sparse_size())
out = conv(node_feats, node_attrs, edge_feats, edge_attrs, graph)
