# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

from cugraph_pyg.nn import RGCNConv as CuGraphRGCNConv

ATOL = 1e-6


@pytest.mark.parametrize("aggr", ["add", "sum", "mean"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("max_num_neighbors", [8, None])
@pytest.mark.parametrize("num_bases", [1, 2, None])
@pytest.mark.parametrize("root_weight", [True, False])
@pytest.mark.parametrize("graph", ["basic_pyg_graph_1", "basic_pyg_graph_2"])
@pytest.mark.sg
def test_rgcn_conv_equality(
    aggr, bias, max_num_neighbors, num_bases, root_weight, graph, request
):
    pytest.importorskip("torch_geometric", reason="PyG not available")
    import torch
    from torch_geometric.nn import FastRGCNConv as RGCNConv

    torch.manual_seed(12345)
    in_channels, out_channels, num_relations = (4, 2, 3)
    kwargs = dict(aggr=aggr, bias=bias, num_bases=num_bases, root_weight=root_weight)

    edge_index, size = request.getfixturevalue(graph)
    edge_index = edge_index.cuda()
    edge_type = torch.randint(num_relations, (edge_index.size(1),)).cuda()

    x = torch.rand(size[0], in_channels, device="cuda")
    csc, edge_type_perm = CuGraphRGCNConv.to_csc(edge_index, size, edge_type)

    conv1 = RGCNConv(in_channels, out_channels, num_relations, **kwargs).cuda()
    conv2 = CuGraphRGCNConv(in_channels, out_channels, num_relations, **kwargs).cuda()

    with torch.no_grad():
        if root_weight:
            conv2.weight.data[:-1] = conv1.weight.data
            conv2.weight.data[-1] = conv1.root.data
        else:
            conv2.weight.data = conv1.weight.data.detach().clone()
        if num_bases is not None:
            conv2.comp.data = conv1.comp.data.detach().clone()

    out1 = conv1(x, edge_index, edge_type)
    out2 = conv2(x, csc, edge_type_perm, max_num_neighbors=max_num_neighbors)
    assert torch.allclose(out1, out2, atol=ATOL)

    grad_out = torch.rand_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)

    if root_weight:
        assert torch.allclose(conv1.weight.grad, conv2.weight.grad[:-1], atol=ATOL)
        assert torch.allclose(conv1.root.grad, conv2.weight.grad[-1], atol=ATOL)
    else:
        assert torch.allclose(conv1.weight.grad, conv2.weight.grad, atol=ATOL)

    if num_bases is not None:
        assert torch.allclose(conv1.comp.grad, conv2.comp.grad, atol=ATOL)
