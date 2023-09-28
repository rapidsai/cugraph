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
from cugraph_dgl.nn import RelGraphConv as CuGraphRelGraphConv
from .common import create_graph1

dgl = pytest.importorskip("dgl", reason="DGL not available")
torch = pytest.importorskip("torch", reason="PyTorch not available")

ATOL = 1e-6


@pytest.mark.parametrize("idtype_int", [False, True])
@pytest.mark.parametrize("max_in_degree", [None, 8])
@pytest.mark.parametrize("num_bases", [1, 2, 5])
@pytest.mark.parametrize("regularizer", [None, "basis"])
@pytest.mark.parametrize("self_loop", [False, True])
@pytest.mark.parametrize("to_block", [False, True])
@pytest.mark.parametrize("sparse_format", ["coo", "csc", None])
def test_relgraphconv_equality(
    idtype_int,
    max_in_degree,
    num_bases,
    regularizer,
    self_loop,
    to_block,
    sparse_format,
):
    from dgl.nn.pytorch import RelGraphConv

    torch.manual_seed(12345)
    in_feat, out_feat, num_rels = 10, 2, 3
    args = (in_feat, out_feat, num_rels)
    kwargs = {
        "num_bases": num_bases,
        "regularizer": regularizer,
        "bias": False,
        "self_loop": self_loop,
    }
    g = create_graph1().to("cuda")
    g.edata[dgl.ETYPE] = torch.randint(num_rels, (g.num_edges(),)).cuda()

    if idtype_int:
        g = g.int()
    if to_block:
        g = dgl.to_block(g)

    size = (g.num_src_nodes(), g.num_dst_nodes())
    feat = torch.rand(g.num_src_nodes(), in_feat).cuda()

    if sparse_format == "coo":
        sg = SparseGraph(
            size=size,
            src_ids=g.edges()[0],
            dst_ids=g.edges()[1],
            values=g.edata[dgl.ETYPE],
            formats="csc",
        )
    elif sparse_format == "csc":
        offsets, indices, perm = g.adj_tensors("csc")
        etypes = g.edata[dgl.ETYPE][perm]
        sg = SparseGraph(
            size=size, src_ids=indices, cdst_ids=offsets, values=etypes, formats="csc"
        )

    conv1 = RelGraphConv(*args, **kwargs).cuda()
    conv2 = CuGraphRelGraphConv(*args, **kwargs, apply_norm=False).cuda()

    with torch.no_grad():
        if self_loop:
            conv2.W.data[:-1] = conv1.linear_r.W.data
            conv2.W.data[-1] = conv1.loop_weight.data
        else:
            conv2.W.data = conv1.linear_r.W.data.detach().clone()

        if regularizer is not None:
            conv2.coeff.data = conv1.linear_r.coeff.data.detach().clone()

    out1 = conv1(g, feat, g.edata[dgl.ETYPE])

    if sparse_format is not None:
        out2 = conv2(sg, feat, sg.values(), max_in_degree=max_in_degree)
    else:
        out2 = conv2(g, feat, g.edata[dgl.ETYPE], max_in_degree=max_in_degree)

    assert torch.allclose(out1, out2, atol=ATOL)

    grad_out = torch.rand_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)

    end = -1 if self_loop else None
    assert torch.allclose(conv1.linear_r.W.grad, conv2.W.grad[:end], atol=ATOL)

    if self_loop:
        assert torch.allclose(conv1.loop_weight.grad, conv2.W.grad[-1], atol=ATOL)

    if regularizer is not None:
        assert torch.allclose(conv1.linear_r.coeff.grad, conv2.coeff.grad, atol=ATOL)
