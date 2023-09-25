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

import pytest

from cugraph_pyg.nn import TransformerConv as CuGraphTransformerConv

ATOL = 1e-6


@pytest.mark.parametrize("bipartite", [True, False])
@pytest.mark.parametrize("concat", [True, False])
@pytest.mark.parametrize("heads", [1, 2, 3, 5, 10, 16])
@pytest.mark.parametrize("graph", ["basic_pyg_graph_1", "basic_pyg_graph_2"])
def test_transformer_conv_equality(bipartite, concat, heads, graph, request):
    pytest.importorskip("torch_geometric", reason="PyG not available")
    import torch
    from torch_geometric.nn import TransformerConv

    torch.manual_seed(12345)
    edge_index, size = request.getfixturevalue(graph)
    edge_index = edge_index.cuda()
    csc = CuGraphTransformerConv.to_csc(edge_index, size)

    out_channels = 2
    kwargs = dict(concat=concat, bias=False, root_weight=False)

    if bipartite:
        in_channels = (5, 3)
        x = (
            torch.rand(size[0], in_channels[0], device="cuda"),
            torch.rand(size[1], in_channels[1], device="cuda"),
        )
    else:
        in_channels = 5
        x = torch.rand(size[0], in_channels, device="cuda")

    conv1 = TransformerConv(in_channels, out_channels, heads, **kwargs).cuda()
    conv2 = CuGraphTransformerConv(in_channels, out_channels, heads, **kwargs).cuda()

    with torch.no_grad():
        conv2.lin_query.weight.data = conv1.lin_query.weight.data.detach().clone()
        conv2.lin_key.weight.data = conv1.lin_key.weight.data.detach().clone()
        conv2.lin_value.weight.data = conv1.lin_value.weight.data.detach().clone()
        conv2.lin_query.bias.data = conv1.lin_query.bias.data.detach().clone()
        conv2.lin_key.bias.data = conv1.lin_key.bias.data.detach().clone()
        conv2.lin_value.bias.data = conv1.lin_value.bias.data.detach().clone()

    out1 = conv1(x, edge_index)
    out2 = conv2(x, csc)

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
