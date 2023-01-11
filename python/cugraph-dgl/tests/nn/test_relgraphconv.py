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
    import cugraph_dgl
except ModuleNotFoundError:
    pytest.skip("cugraph_dgl not available", allow_module_level=True)

from cugraph.utilities.utils import import_optional
from .common import create_graph1

th = import_optional("torch")
dgl = import_optional("dgl")

options = {
    "idtype_int": [False, True],
    "max_in_degree": [None, 8],
    "regularizer": [None, "basis"],
    "to_block": [False, True],
}


@pytest.mark.parametrize("to_block", options["to_block"])
@pytest.mark.parametrize("regularizer", options["regularizer"])
@pytest.mark.parametrize("max_in_degree", options["max_in_degree"])
@pytest.mark.parametrize("idtype_int", options["idtype_int"])
def test_relgraphconv_equality(idtype_int, max_in_degree, regularizer, to_block):
    RelGraphConv = dgl.nn.RelGraphConv
    CuGraphRelGraphConv = cugraph_dgl.nn.RelGraphConv

    device = "cuda"
    in_feat, out_feat, num_rels = 10, 2, 3
    args = (in_feat, out_feat, num_rels)
    kwargs = {
        "num_bases": 2,
        "regularizer": regularizer,
        "bias": False,
        "self_loop": False,
    }
    g = create_graph1().to(device)
    g.edata[dgl.ETYPE] = th.randint(num_rels, (g.num_edges(),)).to(device)
    if idtype_int:
        g = g.int()
    if to_block:
        g = dgl.to_block(g)
    feat = th.rand(g.num_src_nodes(), in_feat).to(device)

    th.manual_seed(0)
    conv1 = RelGraphConv(*args, **kwargs).to(device)

    th.manual_seed(0)
    kwargs["max_in_degree"] = max_in_degree
    kwargs["apply_norm"] = False
    conv2 = CuGraphRelGraphConv(*args, **kwargs).to(device)

    out1 = conv1(g, feat, g.edata[dgl.ETYPE])
    out2 = conv2(g, feat, g.edata[dgl.ETYPE])
    assert th.allclose(out1, out2, atol=1e-06)

    grad_out = th.rand_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)
    assert th.allclose(conv1.linear_r.W.grad, conv2.W.grad, atol=1e-6)
    if regularizer is not None:
        assert th.allclose(conv1.linear_r.coeff.grad, conv2.coeff.grad, atol=1e-6)
