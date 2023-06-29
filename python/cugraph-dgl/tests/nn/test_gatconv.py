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
# pylint: disable=too-many-arguments, too-many-locals

import pytest

try:
    import cugraph_dgl
except ModuleNotFoundError:
    pytest.skip("cugraph_dgl not available", allow_module_level=True)

from cugraph.utilities.utils import import_optional
from .common import create_graph1

torch = import_optional("torch")
dgl = import_optional("dgl")


@pytest.mark.parametrize("bipartite", [False, True])
@pytest.mark.parametrize("idtype_int", [False, True])
@pytest.mark.parametrize("max_in_degree", [None, 8])
@pytest.mark.parametrize("num_heads", [1, 2, 7])
@pytest.mark.parametrize("to_block", [False, True])
def test_gatconv_equality(bipartite, idtype_int, max_in_degree, num_heads, to_block):
    GATConv = dgl.nn.GATConv
    CuGraphGATConv = cugraph_dgl.nn.GATConv
    device = "cuda"
    g = create_graph1().to(device)

    if idtype_int:
        g = g.int()

    if to_block:
        g = dgl.to_block(g)

    if bipartite:
        in_feats = (10, 3)
        nfeat = (
            torch.rand(g.num_src_nodes(), in_feats[0], device=device),
            torch.rand(g.num_dst_nodes(), in_feats[1], device=device),
        )
    else:
        in_feats = 10
        nfeat = torch.rand(g.num_src_nodes(), in_feats, device=device)
    out_feats = 2

    args = (in_feats, out_feats, num_heads)
    kwargs = {"bias": False}

    conv1 = GATConv(*args, **kwargs, allow_zero_in_degree=True).to(device)
    out1 = conv1(g, nfeat)

    conv2 = CuGraphGATConv(*args, **kwargs).to(device)
    dim = num_heads * out_feats
    with torch.no_grad():
        conv2.attn_weights.data[:dim] = conv1.attn_l.data.flatten()
        conv2.attn_weights.data[dim:] = conv1.attn_r.data.flatten()
        if bipartite:
            conv2.fc_src.weight.data = conv1.fc_src.weight.data.detach().clone()
            conv2.fc_dst.weight.data = conv1.fc_dst.weight.data.detach().clone()
        else:
            conv2.fc.weight.data = conv1.fc.weight.data.detach().clone()
    out2 = conv2(g, nfeat, max_in_degree=max_in_degree)

    assert torch.allclose(out1, out2, atol=1e-6)

    grad_out1 = torch.rand_like(out1)
    grad_out2 = grad_out1.clone().detach()
    out1.backward(grad_out1)
    out2.backward(grad_out2)

    if bipartite:
        assert torch.allclose(
            conv1.fc_src.weight.grad, conv2.fc_src.weight.grad, atol=1e-6
        )
        assert torch.allclose(
            conv1.fc_dst.weight.grad, conv2.fc_dst.weight.grad, atol=1e-6
        )
    else:
        assert torch.allclose(conv1.fc.weight.grad, conv2.fc.weight.grad, atol=1e-6)

    assert torch.allclose(
        torch.cat((conv1.attn_l.grad, conv1.attn_r.grad), dim=0),
        conv2.attn_weights.grad.view(2, num_heads, out_feats),
        atol=1e-6,
    )


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("bipartite", [False, True])
@pytest.mark.parametrize("concat", [False, True])
@pytest.mark.parametrize("max_in_degree", [None, 8, 800])
@pytest.mark.parametrize("num_heads", [1, 2, 7])
@pytest.mark.parametrize("to_block", [False, True])
@pytest.mark.parametrize("use_edge_feats", [False, True])
def test_gatconv_edge_feats(
    bias, bipartite, concat, max_in_degree, num_heads, to_block, use_edge_feats
):
    from cugraph_dgl.nn import GATConv

    device = "cuda"
    g = create_graph1().to(device)

    if to_block:
        g = dgl.to_block(g)

    if bipartite:
        in_feats = (10, 3)
        nfeat = (
            torch.rand(g.num_src_nodes(), in_feats[0], device=device),
            torch.rand(g.num_dst_nodes(), in_feats[1], device=device),
        )
    else:
        in_feats = 10
        nfeat = torch.rand(g.num_src_nodes(), in_feats, device=device)
    out_feats = 2

    if use_edge_feats:
        edge_feats = 3
        efeat = torch.rand(g.num_edges(), edge_feats, device=device)
    else:
        edge_feats = None
        efeat = None

    conv = GATConv(
        in_feats, out_feats, num_heads, concat=concat, edge_feats=edge_feats, bias=bias
    ).to(device)
    out = conv(g, nfeat, efeat=efeat, max_in_degree=max_in_degree)

    grad_out = torch.rand_like(out)
    out.backward(grad_out)
