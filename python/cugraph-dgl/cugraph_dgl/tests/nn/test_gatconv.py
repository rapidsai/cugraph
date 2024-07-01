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

from cugraph_dgl.nn.conv.base import SparseGraph
from cugraph_dgl.nn import GATConv as CuGraphGATConv

dgl = pytest.importorskip("dgl", reason="DGL not available")
torch = pytest.importorskip("torch", reason="PyTorch not available")

ATOL = 1e-6


@pytest.mark.parametrize("mode", ["bipartite", "share_weights", "regular"])
@pytest.mark.parametrize("idx_type", [torch.int32, torch.int64])
@pytest.mark.parametrize("max_in_degree", [None, 8])
@pytest.mark.parametrize("num_heads", [1, 2, 7])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("to_block", [False, True])
@pytest.mark.parametrize("sparse_format", ["coo", "csc", None])
def test_gatconv_equality(
    dgl_graph_1,
    mode,
    idx_type,
    max_in_degree,
    num_heads,
    residual,
    to_block,
    sparse_format,
):
    from dgl.nn.pytorch import GATConv

    torch.manual_seed(12345)
    device = torch.device("cuda")
    g = dgl_graph_1.to(device).astype(idx_type)

    if to_block:
        g = dgl.to_block(g)

    size = (g.num_src_nodes(), g.num_dst_nodes())

    if mode == "bipartite":
        in_feats = (10, 3)
        nfeat = (
            torch.randn(size[0], in_feats[0]).to(device),
            torch.randn(size[1], in_feats[1]).to(device),
        )
    elif mode == "share_weights":
        in_feats = 5
        nfeat = (
            torch.randn(size[0], in_feats).to(device),
            torch.randn(size[1], in_feats).to(device),
        )
    else:
        in_feats = 7
        nfeat = torch.randn(size[0], in_feats).to(device)
    out_feats = 2

    if sparse_format == "coo":
        sg = SparseGraph(
            size=size, src_ids=g.edges()[0], dst_ids=g.edges()[1], formats="csc"
        )
    elif sparse_format == "csc":
        offsets, indices, _ = g.adj_tensors("csc")
        sg = SparseGraph(size=size, src_ids=indices, cdst_ids=offsets, formats="csc")

    args = (in_feats, out_feats, num_heads)
    kwargs = {"bias": False, "allow_zero_in_degree": True, "residual": residual}

    conv1 = GATConv(*args, **kwargs).to(device)
    conv2 = CuGraphGATConv(*args, **kwargs).to(device)

    dim = num_heads * out_feats
    with torch.no_grad():
        conv2.attn_weights[:dim].copy_(conv1.attn_l.flatten())
        conv2.attn_weights[dim:].copy_(conv1.attn_r.flatten())
        if mode == "bipartite":
            conv2.lin_src.weight.copy_(conv1.fc_src.weight)
            conv2.lin_dst.weight.copy_(conv1.fc_dst.weight)
        else:
            conv2.lin.weight.copy_(conv1.fc.weight)
        if residual and conv1.has_linear_res:
            conv2.lin_res.weight.copy_(conv1.res_fc.weight)

    out1 = conv1(g, nfeat)
    if sparse_format is not None:
        out2 = conv2(sg, nfeat, max_in_degree=max_in_degree)
    else:
        out2 = conv2(g, nfeat, max_in_degree=max_in_degree)

    assert torch.allclose(out1, out2, atol=ATOL)

    grad_out1 = torch.randn_like(out1)
    grad_out2 = grad_out1.detach().clone()
    out1.backward(grad_out1)
    out2.backward(grad_out2)

    if mode == "bipartite":
        assert torch.allclose(
            conv1.fc_src.weight.grad, conv2.lin_src.weight.grad, atol=ATOL
        )
        assert torch.allclose(
            conv1.fc_dst.weight.grad, conv2.lin_dst.weight.grad, atol=ATOL
        )
    else:
        assert torch.allclose(conv1.fc.weight.grad, conv2.lin.weight.grad, atol=ATOL)

    if residual and conv1.has_linear_res:
        assert torch.allclose(
            conv1.res_fc.weight.grad, conv2.lin_res.weight.grad, atol=ATOL
        )

    assert torch.allclose(
        torch.cat((conv1.attn_l.grad, conv1.attn_r.grad), dim=0),
        conv2.attn_weights.grad.view(2, num_heads, out_feats),
        atol=1e-5,  # Note: using a loosened tolerance here due to numerical error
    )


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("bipartite", [False, True])
@pytest.mark.parametrize("concat", [False, True])
@pytest.mark.parametrize("max_in_degree", [None, 8])
@pytest.mark.parametrize("num_heads", [1, 2, 7])
@pytest.mark.parametrize("to_block", [False, True])
@pytest.mark.parametrize("use_edge_feats", [False, True])
def test_gatconv_edge_feats(
    dgl_graph_1,
    bias,
    bipartite,
    concat,
    max_in_degree,
    num_heads,
    to_block,
    use_edge_feats,
):
    torch.manual_seed(12345)
    device = torch.device("cuda")
    g = dgl_graph_1.to(device)

    if to_block:
        g = dgl.to_block(g)

    if bipartite:
        in_feats = (10, 3)
        nfeat = (
            torch.rand(g.num_src_nodes(), in_feats[0]).to(device),
            torch.rand(g.num_dst_nodes(), in_feats[1]).to(device),
        )
    else:
        in_feats = 10
        nfeat = torch.rand(g.num_src_nodes(), in_feats).to(device)
    out_feats = 2

    if use_edge_feats:
        edge_feats = 3
        efeat = torch.rand(g.num_edges(), edge_feats).to(device)
    else:
        edge_feats = None
        efeat = None

    conv = CuGraphGATConv(
        in_feats,
        out_feats,
        num_heads,
        concat=concat,
        edge_feats=edge_feats,
        bias=bias,
        allow_zero_in_degree=True,
    ).to(device)
    out = conv(g, nfeat, efeat=efeat, max_in_degree=max_in_degree)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
