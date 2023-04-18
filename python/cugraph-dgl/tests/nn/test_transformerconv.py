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
from itertools import product
import pytest

try:
    from cugraph_dgl.nn import TransformerConv
except ModuleNotFoundError:
    pytest.skip("cugraph_dgl not available", allow_module_level=True)

from cugraph.utilities.utils import import_optional
from .common import create_graph1

torch = import_optional("torch")
dgl = import_optional("dgl")


def test_TransformerConv():
    device = "cuda"
    in_node_feats = (5, 3)
    out_node_feats = 2
    edge_feats = 3
    num_heads = 4
    concat = True

    g = create_graph1().to(device)

    nfeats = (
        torch.rand(g.num_src_nodes(), in_node_feats[0], device=device),
        torch.rand(g.num_dst_nodes(), in_node_feats[1], device=device),
    )

    efeats = torch.rand(g.num_edges(), edge_feats, device=device)

    conv = TransformerConv(
        in_node_feats,
        out_node_feats,
        num_heads=num_heads,
        concat=concat,
        edge_feats=edge_feats,
    ).to(device)

    out = conv(g, nfeats, efeats)
