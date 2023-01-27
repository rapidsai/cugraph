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
    "num_bases": [1, 2, 5],
    "regularizer": [None, "basis"],
    "self_loop": [False, True],
    "to_block": [False, True],
}


@pytest.mark.parametrize(",".join(options.keys()), product(*options.values()))
def test_relgraphconv_equality(
    idtype_int, max_in_degree, num_bases, regularizer, self_loop, to_block
):
    RelGraphConv = dgl.nn.RelGraphConv
    CuGraphRelGraphConv = cugraph_dgl.nn.RelGraphConv
    device = "cuda"

    in_feat, out_feat, num_rels = 10, 2, 3
    args = (in_feat, out_feat, num_rels)
    kwargs = {
        "num_bases": num_bases,
        "regularizer": regularizer,
        "bias": False,
        "self_loop": self_loop,
    }
    g = create_graph1().to(device)
    g.edata[dgl.ETYPE] = torch.randint(num_rels, (g.num_edges(),)).to(device)
    if idtype_int:
        g = g.int()
    if to_block:
        g = dgl.to_block(g)
    feat = torch.rand(g.num_src_nodes(), in_feat).to(device)

    torch.manual_seed(0)
    conv1 = RelGraphConv(*args, **kwargs).to(device)

    torch.manual_seed(0)
    kwargs["apply_norm"] = False
    conv2 = CuGraphRelGraphConv(*args, **kwargs).to(device)

    out1 = conv1(g, feat, g.edata[dgl.ETYPE])
    out2 = conv2(g, feat, g.edata[dgl.ETYPE], max_in_degree=max_in_degree)
    assert torch.allclose(out1, out2, atol=1e-06)

    grad_out = torch.rand_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)

    end = -1 if self_loop else None
    assert torch.allclose(conv1.linear_r.W.grad, conv2.W.grad[:end], atol=1e-6)

    if self_loop:
        assert torch.allclose(conv1.loop_weight.grad, conv2.W.grad[-1], atol=1e-6)

    if regularizer is not None:
        assert torch.allclose(conv1.linear_r.coeff.grad, conv2.coeff.grad, atol=1e-6)
