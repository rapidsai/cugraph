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
from cugraph_dgl.nn import SAGEConv as CuGraphSAGEConv

dgl = pytest.importorskip("dgl", reason="DGL not available")
torch = pytest.importorskip("torch", reason="PyTorch not available")

ATOL = 1e-6


@pytest.mark.parametrize("aggr", ["mean", "pool"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("bipartite", [False, True])
@pytest.mark.parametrize("idx_type", [torch.int32, torch.int64])
@pytest.mark.parametrize("max_in_degree", [None, 8])
@pytest.mark.parametrize("to_block", [False, True])
@pytest.mark.parametrize("sparse_format", ["coo", "csc", None])
def test_sageconv_equality(
    dgl_graph_1, aggr, bias, bipartite, idx_type, max_in_degree, to_block, sparse_format
):
    from dgl.nn.pytorch import SAGEConv

    torch.manual_seed(12345)
    device = torch.device("cuda:0")
    g = dgl_graph_1.to(device).astype(idx_type)

    if to_block:
        g = dgl.to_block(g)

    size = (g.num_src_nodes(), g.num_dst_nodes())

    if bipartite:
        in_feats = (5, 3)
        feat = (
            torch.rand(size[0], in_feats[0], requires_grad=True).to(device),
            torch.rand(size[1], in_feats[1], requires_grad=True).to(device),
        )
    else:
        in_feats = 5
        feat = torch.rand(size[0], in_feats).to(device)
    out_feats = 2

    if sparse_format == "coo":
        sg = SparseGraph(
            size=size, src_ids=g.edges()[0], dst_ids=g.edges()[1], formats="csc"
        )
    elif sparse_format == "csc":
        offsets, indices, _ = g.adj_tensors("csc")
        sg = SparseGraph(size=size, src_ids=indices, cdst_ids=offsets, formats="csc")

    kwargs = {"aggregator_type": aggr, "bias": bias}
    conv1 = SAGEConv(in_feats, out_feats, **kwargs).to(device)
    conv2 = CuGraphSAGEConv(in_feats, out_feats, **kwargs).to(device)

    in_feats_src = conv2.in_feats_src
    with torch.no_grad():
        conv2.lin.weight[:, :in_feats_src].copy_(conv1.fc_neigh.weight)
        conv2.lin.weight[:, in_feats_src:].copy_(conv1.fc_self.weight)
        if bias:
            conv2.lin.bias.copy_(conv1.fc_self.bias)
        if aggr == "pool":
            conv2.pre_lin.weight.copy_(conv1.fc_pool.weight)
            conv2.pre_lin.bias.copy_(conv1.fc_pool.bias)

    out1 = conv1(g, feat)
    if sparse_format is not None:
        out2 = conv2(sg, feat, max_in_degree=max_in_degree)
    else:
        out2 = conv2(g, feat, max_in_degree=max_in_degree)
    assert torch.allclose(out1, out2, atol=ATOL)

    grad_out = torch.randn_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)
    assert torch.allclose(
        conv1.fc_neigh.weight.grad,
        conv2.lin.weight.grad[:, :in_feats_src],
        atol=ATOL,
    )
    assert torch.allclose(
        conv1.fc_self.weight.grad,
        conv2.lin.weight.grad[:, in_feats_src:],
        atol=ATOL,
    )
    if bias:
        assert torch.allclose(conv1.fc_self.bias.grad, conv2.lin.bias.grad, atol=ATOL)
