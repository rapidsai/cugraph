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
    from cugraph_dgl.nn import TransformerConv
except ModuleNotFoundError:
    pytest.skip("cugraph_dgl not available", allow_module_level=True)

from cugraph.utilities.utils import import_optional
from .common import create_graph1

torch = import_optional("torch")
dgl = import_optional("dgl")


@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("bipartite_node_feats", [False, True])
@pytest.mark.parametrize("concat", [False, True])
@pytest.mark.parametrize("idtype_int", [False, True])
@pytest.mark.parametrize("num_heads", [1, 2, 3, 4])
@pytest.mark.parametrize("to_block", [False, True])
@pytest.mark.parametrize("use_edge_feats", [False, True])
def test_TransformerConv(
    beta, bipartite_node_feats, concat, idtype_int, num_heads, to_block, use_edge_feats
):
    device = "cuda"
    g = create_graph1().to(device)

    if idtype_int:
        g = g.int()

    if to_block:
        g = dgl.to_block(g)

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

    out = conv(g, nfeat, efeat)
    grad_out = torch.rand_like(out)
    out.backward(grad_out)
