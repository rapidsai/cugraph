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

from cugraph_pyg.nn import TransformerConv as CuGraphTransformerConv

ATOL = 1e-6


@pytest.mark.parametrize("use_edge_index", [True, False])
@pytest.mark.parametrize("use_edge_attr", [True, False])
@pytest.mark.parametrize("bipartite", [True, False])
@pytest.mark.parametrize("concat", [True, False])
@pytest.mark.parametrize("heads", [1, 2, 3, 5, 10, 16])
@pytest.mark.parametrize("graph", ["basic_pyg_graph_1", "basic_pyg_graph_2"])
def test_transformer_conv_equality(
    use_edge_index, use_edge_attr, bipartite, concat, heads, graph, request
):
    pytest.importorskip("torch_geometric", reason="PyG not available")
    import torch
    from torch_geometric import EdgeIndex
    from torch_geometric.nn import TransformerConv

    torch.manual_seed(12345)
    edge_index, size = request.getfixturevalue(graph)
    edge_index = edge_index.cuda()

    if bipartite:
        in_channels = (5, 3)
        x = (
            torch.rand(size[0], in_channels[0], device="cuda"),
            torch.rand(size[1], in_channels[1], device="cuda"),
        )
    else:
        in_channels = 5
        x = torch.rand(size[0], in_channels, device="cuda")
    out_channels = 2

    if use_edge_attr:
        edge_dim = 3
        edge_attr = torch.rand(edge_index.size(1), edge_dim).cuda()
    else:
        edge_dim = edge_attr = None

    if use_edge_index:
        csc = EdgeIndex(edge_index, sparse_size=size)
    else:
        if use_edge_attr:
            csc, edge_attr_perm = CuGraphTransformerConv.to_csc(
                edge_index, size, edge_attr=edge_attr
            )
        else:
            csc = CuGraphTransformerConv.to_csc(edge_index, size)
            edge_attr_perm = None

    kwargs = dict(concat=concat, bias=False, edge_dim=edge_dim, root_weight=False)

    conv1 = TransformerConv(in_channels, out_channels, heads, **kwargs).cuda()
    conv2 = CuGraphTransformerConv(in_channels, out_channels, heads, **kwargs).cuda()

    with torch.no_grad():
        conv2.lin_query.weight.copy_(conv1.lin_query.weight)
        conv2.lin_key.weight.copy_(conv1.lin_key.weight)
        conv2.lin_value.weight.copy_(conv1.lin_value.weight)
        conv2.lin_query.bias.copy_(conv1.lin_query.bias)
        conv2.lin_key.bias.copy_(conv1.lin_key.bias)
        conv2.lin_value.bias.copy_(conv1.lin_value.bias)
        if use_edge_attr:
            conv2.lin_edge.weight.copy_(conv1.lin_edge.weight)

    out1 = conv1(x, edge_index, edge_attr=edge_attr)
    if use_edge_index:
        out2 = conv2(x, csc, edge_attr=edge_attr)
    else:
        out2 = conv2(x, csc, edge_attr=edge_attr_perm)

    assert torch.allclose(out1, out2, atol=ATOL)

    grad_output = torch.rand_like(out1)
    out1.backward(grad_output)
    out2.backward(grad_output)

    assert torch.allclose(
        conv1.lin_query.weight.grad, conv2.lin_query.weight.grad, atol=ATOL
    )
    assert torch.allclose(
        conv1.lin_key.weight.grad, conv2.lin_key.weight.grad, atol=ATOL
    )
    assert torch.allclose(
        conv1.lin_value.weight.grad, conv2.lin_value.weight.grad, atol=ATOL
    )
    assert torch.allclose(
        conv1.lin_query.bias.grad, conv2.lin_query.bias.grad, atol=ATOL
    )
    assert torch.allclose(conv1.lin_key.bias.grad, conv2.lin_key.bias.grad, atol=ATOL)
    assert torch.allclose(
        conv1.lin_value.bias.grad, conv2.lin_value.bias.grad, atol=ATOL
    )

    if use_edge_attr:
        assert torch.allclose(
            conv1.lin_edge.weight.grad, conv2.lin_edge.weight.grad, atol=ATOL
        )
