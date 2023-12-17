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

from cugraph_dgl.nn.conv.base import SparseGraph
from cugraph_dgl.nn import TransformerConv

dgl = pytest.importorskip("dgl", reason="DGL not available")
torch = pytest.importorskip("torch", reason="PyTorch not available")

ATOL = 1e-6


@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("bipartite_node_feats", [False, True])
@pytest.mark.parametrize("concat", [False, True])
@pytest.mark.parametrize("idx_type", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_heads", [1, 3, 4])
@pytest.mark.parametrize("to_block", [False, True])
@pytest.mark.parametrize("use_edge_feats", [False, True])
@pytest.mark.parametrize("sparse_format", ["coo", "csc", None])
def test_transformerconv(
    dgl_graph_1,
    beta,
    bipartite_node_feats,
    concat,
    idx_type,
    num_heads,
    to_block,
    use_edge_feats,
    sparse_format,
):
    torch.manual_seed(12345)
    device = torch.device("cuda:0")
    g = dgl_graph_1.to(device).astype(idx_type)

    if to_block:
        g = dgl.to_block(g)

    size = (g.num_src_nodes(), g.num_dst_nodes())
    if sparse_format == "coo":
        sg = SparseGraph(
            size=size, src_ids=g.edges()[0], dst_ids=g.edges()[1], formats="csc"
        )
    elif sparse_format == "csc":
        offsets, indices, _ = g.adj_tensors("csc")
        sg = SparseGraph(size=size, src_ids=indices, cdst_ids=offsets, formats="csc")

    if bipartite_node_feats:
        in_node_feats = (5, 3)
        nfeat = (
            torch.rand(g.num_src_nodes(), in_node_feats[0], device=device),
            torch.rand(g.num_dst_nodes(), in_node_feats[1], device=device),
        )
    else:
        in_node_feats = 3
        nfeat = torch.rand(g.num_src_nodes(), in_node_feats, device=device)
    out_node_feats = 2

    if use_edge_feats:
        edge_feats = 3
        efeat = torch.rand(g.num_edges(), edge_feats, device=device)
    else:
        edge_feats = None
        efeat = None

    conv = TransformerConv(
        in_node_feats,
        out_node_feats,
        num_heads=num_heads,
        concat=concat,
        beta=beta,
        edge_feats=edge_feats,
    ).to(device)

    if sparse_format is not None:
        out = conv(sg, nfeat, efeat)
    else:
        out = conv(g, nfeat, efeat)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
