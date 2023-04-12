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

try:
    from torch_geometric.nn import TransformerConv
except ModuleNotFoundError:
    pytest.skip("PyG not available", allow_module_level=True)

from cugraph.utilities.utils import import_optional
from cugraph_pyg.nn import TransformerConv as CuGraphTransformerConv

torch = import_optional("torch")

@pytest.mark.parametrize("bipartite", [False])
@pytest.mark.parametrize("concat", [True])
@pytest.mark.parametrize("heads", [1, 2, 3])
@pytest.mark.parametrize("max_num_neighbors", [8, None])
def test_transformer_conv_equality(bipartite, concat, heads, max_num_neighbors):
    in_channels, out_channels = (5, 2)
    kwargs = dict(concat=concat)

    size = (10, 8) if bipartite else (10, 10)
    x = torch.rand(size[0], in_channels, device="cuda")
    edge_index = torch.tensor(
        [
            [7, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7],
        ],
        device="cuda",
    )

    conv1 = TransformerConv(in_channels, out_channels, heads, bias=False, root_weight=False, **kwargs).cuda()
    conv2 = CuGraphTransformerConv(in_channels, out_channels, heads, **kwargs).cuda()

    with torch.no_grad():
        conv2.lin_query.weight.data[:, :] = conv1.lin_query.weight.data
        conv2.lin_key.weight.data[:, :] = conv1.lin_key.weight.data
        conv2.lin_value.weight.data[:, :] = conv1.lin_value.weight.data
        conv2.lin_query.bias.data[:] = conv1.lin_query.bias.data
        conv2.lin_key.bias.data[:] = conv1.lin_key.bias.data
        conv2.lin_value.bias.data[:] = conv1.lin_value.bias.data

    if bipartite:
        out1 = conv1((x, x[: size[1]]), edge_index)
    else:
        out1 = conv1(x, edge_index)

    csc = CuGraphTransformerConv.to_csc(edge_index, size)
    out2 = conv2(x, csc, max_num_neighbors=max_num_neighbors)
    assert torch.allclose(out1, out2, atol=1e-2)

    grad_output = torch.rand_like(out1)
    out1.backward(grad_output)
    out2.backward(grad_output)

    assert torch.allclose(conv1.lin_query.weight.grad, conv2.lin_query.weight.grad, atol=1e-3)
    assert torch.allclose(conv1.lin_key.weight.grad, conv2.lin_key.weight.grad, atol=1e-3)
    assert torch.allclose(conv1.lin_value.weight.grad, conv2.lin_value.weight.grad, atol=1e-3)
    assert torch.allclose(conv1.lin_query.bias.grad, conv2.lin_query.bias.grad, atol=1e-3)
    assert torch.allclose(conv1.lin_key.bias.grad, conv2.lin_key.bias.grad, atol=1e-3)
    assert torch.allclose(conv1.lin_value.bias.grad, conv2.lin_value.bias.grad, atol=1e-3)
