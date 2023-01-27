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
from itertools import product
import pytest

try:
    import cugraph_dgl
except ModuleNotFoundError:
    pytest.skip("cugraph_dgl not available", allow_module_level=True)

from cugraph.utilities.utils import import_optional
from .common import create_graph1

torch = import_optional("torch")
dgl = import_optional("dgl")

options = {
    "idtype_int": [False, True],
    "max_in_degree": [None, 8],
    "num_heads": [1, 3],
    "to_block": [False, True],
}


@pytest.mark.parametrize(",".join(options.keys()), product(*options.values()))
def test_gatconv_equality(idtype_int, max_in_degree, num_heads, to_block):
    GATConv = dgl.nn.GATConv
    CuGraphGATConv = cugraph_dgl.nn.GATConv
    device = "cuda"

    in_feat, out_feat = 10, 2
    args = (in_feat, out_feat, num_heads)
    kwargs = {"bias": False}
    g = create_graph1().to(device)
    if idtype_int:
        g = g.int()
    if to_block:
        g = dgl.to_block(g)
    feat = torch.rand(g.num_src_nodes(), in_feat).to(device)

    torch.manual_seed(0)
    conv1 = GATConv(*args, **kwargs, allow_zero_in_degree=True).to(device)
    out1 = conv1(g, feat)

    torch.manual_seed(0)
    conv2 = CuGraphGATConv(*args, **kwargs).to(device)
    dim = num_heads * out_feat
    with torch.no_grad():
        conv2.attn_weights.data[:dim] = conv1.attn_l.data.flatten()
        conv2.attn_weights.data[dim:] = conv1.attn_r.data.flatten()
        conv2.fc.weight.data[:] = conv1.fc.weight.data
    out2 = conv2(g, feat, max_in_degree=max_in_degree)
    assert torch.allclose(out1, out2, atol=1e-6)

    grad_out1 = torch.rand_like(out1)
    grad_out2 = grad_out1.clone().detach()
    out1.backward(grad_out1)
    out2.backward(grad_out2)

    assert torch.allclose(conv1.fc.weight.grad, conv2.fc.weight.grad, atol=1e-6)
    assert torch.allclose(
        torch.cat((conv1.attn_l.grad, conv1.attn_r.grad), dim=0),
        conv2.attn_weights.grad.view(2, num_heads, out_feat),
        atol=1e-6,
    )
