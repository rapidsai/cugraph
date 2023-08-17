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

from cugraph.utilities.utils import import_optional
from cugraph_dgl.nn.conv.base import SparseGraph
from cugraph_dgl.nn import SAGEConv as CuGraphSAGEConv
from .common import create_graph1

torch = import_optional("torch")
dgl = import_optional("dgl")


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("idtype_int", [False, True])
@pytest.mark.parametrize("max_in_degree", [None, 8])
@pytest.mark.parametrize("to_block", [False, True])
@pytest.mark.parametrize("sparse_format", ["coo", "csc", None])
def test_SAGEConv_equality(bias, idtype_int, max_in_degree, to_block, sparse_format):
    SAGEConv = dgl.nn.SAGEConv
    device = "cuda"

    in_feat, out_feat = 5, 2
    kwargs = {"aggregator_type": "mean", "bias": bias}
    g = create_graph1().to(device)

    if idtype_int:
        g = g.int()
    if to_block:
        g = dgl.to_block(g)

    size = (g.num_src_nodes(), g.num_dst_nodes())
    feat = torch.rand(g.num_src_nodes(), in_feat).to(device)

    if sparse_format == "coo":
        sg = SparseGraph(
            size=size, src_ids=g.edges()[0], dst_ids=g.edges()[1], formats="csc"
        )
    elif sparse_format == "csc":
        offsets, indices, _ = g.adj_tensors("csc")
        sg = SparseGraph(size=size, src_ids=indices, cdst_ids=offsets, formats="csc")

    torch.manual_seed(0)
    conv1 = SAGEConv(in_feat, out_feat, **kwargs).to(device)

    torch.manual_seed(0)
    conv2 = CuGraphSAGEConv(in_feat, out_feat, **kwargs).to(device)

    with torch.no_grad():
        conv2.linear.weight.data[:, :in_feat] = conv1.fc_neigh.weight.data
        conv2.linear.weight.data[:, in_feat:] = conv1.fc_self.weight.data
        if bias:
            conv2.linear.bias.data[:] = conv1.fc_self.bias.data

    out1 = conv1(g, feat)
    if sparse_format is not None:
        out2 = conv2(sg, feat, max_in_degree=max_in_degree)
    else:
        out2 = conv2(g, feat, max_in_degree=max_in_degree)
    assert torch.allclose(out1, out2, atol=1e-06)

    grad_out = torch.rand_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)
    assert torch.allclose(
        conv1.fc_neigh.weight.grad,
        conv2.linear.weight.grad[:, :in_feat],
        atol=1e-6,
    )
    assert torch.allclose(
        conv1.fc_self.weight.grad,
        conv2.linear.weight.grad[:, in_feat:],
        atol=1e-6,
    )
    if bias:
        assert torch.allclose(
            conv1.fc_self.bias.grad, conv2.linear.bias.grad, atol=1e-6
        )
