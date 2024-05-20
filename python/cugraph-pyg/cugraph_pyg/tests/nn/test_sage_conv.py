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

from cugraph_pyg.nn import SAGEConv as CuGraphSAGEConv

ATOL = 1e-6


@pytest.mark.parametrize("use_edge_index", [True, False])
@pytest.mark.parametrize("aggr", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("bipartite", [True, False])
@pytest.mark.parametrize("max_num_neighbors", [8, None])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("root_weight", [True, False])
@pytest.mark.parametrize("graph", ["basic_pyg_graph_1", "basic_pyg_graph_2"])
def test_sage_conv_equality(
    use_edge_index,
    aggr,
    bias,
    bipartite,
    max_num_neighbors,
    normalize,
    root_weight,
    graph,
    request,
):
    pytest.importorskip("torch_geometric", reason="PyG not available")
    import torch
    from torch_geometric import EdgeIndex
    from torch_geometric.nn import SAGEConv

    torch.manual_seed(12345)
    edge_index, size = request.getfixturevalue(graph)
    edge_index = edge_index.cuda()

    if use_edge_index:
        csc = EdgeIndex(edge_index, sparse_size=size)
    else:
        csc = CuGraphSAGEConv.to_csc(edge_index, size)

    if bipartite:
        in_channels = (7, 3)
        x = (
            torch.rand(size[0], in_channels[0]).cuda(),
            torch.rand(size[1], in_channels[1]).cuda(),
        )
    else:
        in_channels = 5
        x = torch.rand(size[0], in_channels).cuda()
    out_channels = 4

    kwargs = dict(aggr=aggr, bias=bias, normalize=normalize, root_weight=root_weight)

    conv1 = SAGEConv(in_channels, out_channels, **kwargs).cuda()
    conv2 = CuGraphSAGEConv(in_channels, out_channels, **kwargs).cuda()

    in_channels_src = conv2.in_channels_src
    with torch.no_grad():
        conv2.lin.weight[:, :in_channels_src].copy_(conv1.lin_l.weight)
        if root_weight:
            conv2.lin.weight[:, in_channels_src:].copy_(conv1.lin_r.weight)
        if bias:
            conv2.lin.bias.copy_(conv1.lin_l.bias)

    out1 = conv1(x, edge_index)
    out2 = conv2(x, csc, max_num_neighbors=max_num_neighbors)
    assert torch.allclose(out1, out2, atol=ATOL)

    grad_out = torch.rand_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)

    assert torch.allclose(
        conv1.lin_l.weight.grad,
        conv2.lin.weight.grad[:, :in_channels_src],
        atol=ATOL,
    )

    if root_weight:
        assert torch.allclose(
            conv1.lin_r.weight.grad,
            conv2.lin.weight.grad[:, in_channels_src:],
            atol=ATOL,
        )

    if bias:
        assert torch.allclose(
            conv1.lin_l.bias.grad,
            conv2.lin.bias.grad,
            atol=ATOL,
        )
