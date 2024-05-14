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

from cugraph_pyg.nn import GATv2Conv as CuGraphGATv2Conv

ATOL = 1e-6


@pytest.mark.parametrize("bipartite", [True, False])
@pytest.mark.parametrize("concat", [True, False])
@pytest.mark.parametrize("heads", [1, 2, 3, 5, 10, 16])
@pytest.mark.parametrize("use_edge_attr", [True, False])
@pytest.mark.parametrize("graph", ["basic_pyg_graph_1", "basic_pyg_graph_2"])
@pytest.mark.sg
def test_gatv2_conv_equality(bipartite, concat, heads, use_edge_attr, graph, request):
    pytest.importorskip("torch_geometric", reason="PyG not available")
    import torch
    from torch_geometric.nn import GATv2Conv

    torch.manual_seed(12345)
    edge_index, size = request.getfixturevalue(graph)
    edge_index = edge_index.cuda()

    if bipartite:
        in_channels = (5, 3)
        x = (
            torch.rand(size[0], in_channels[0]).cuda(),
            torch.rand(size[1], in_channels[1]).cuda(),
        )
    else:
        in_channels = 5
        x = torch.rand(size[0], in_channels).cuda()
    out_channels = 2

    if use_edge_attr:
        edge_dim = 3
        edge_attr = torch.rand(edge_index.size(1), edge_dim).cuda()
        csc, edge_attr_perm = CuGraphGATv2Conv.to_csc(
            edge_index, size, edge_attr=edge_attr
        )
    else:
        edge_dim = None
        edge_attr = edge_attr_perm = None
        csc = CuGraphGATv2Conv.to_csc(edge_index, size)

    kwargs = dict(bias=False, concat=concat, edge_dim=edge_dim)

    conv1 = GATv2Conv(
        in_channels, out_channels, heads, add_self_loops=False, **kwargs
    ).cuda()
    conv2 = CuGraphGATv2Conv(in_channels, out_channels, heads, **kwargs).cuda()

    with torch.no_grad():
        conv2.lin_src.weight.data = conv1.lin_l.weight.data.detach().clone()
        conv2.lin_dst.weight.data = conv1.lin_r.weight.data.detach().clone()
        conv2.att.data = conv1.att.data.flatten().detach().clone()
        if use_edge_attr:
            conv2.lin_edge.weight.data = conv1.lin_edge.weight.data.detach().clone()

    out1 = conv1(x, edge_index, edge_attr=edge_attr)
    out2 = conv2(x, csc, edge_attr=edge_attr_perm)
    assert torch.allclose(out1, out2, atol=ATOL)

    grad_output = torch.rand_like(out1)
    out1.backward(grad_output)
    out2.backward(grad_output)

    assert torch.allclose(conv1.lin_l.weight.grad, conv2.lin_src.weight.grad, atol=ATOL)
    assert torch.allclose(conv1.lin_r.weight.grad, conv2.lin_dst.weight.grad, atol=ATOL)

    assert torch.allclose(conv1.att.grad.flatten(), conv2.att.grad, atol=ATOL)

    if use_edge_attr:
        assert torch.allclose(
            conv1.lin_edge.weight.grad, conv2.lin_edge.weight.grad, atol=ATOL
        )
