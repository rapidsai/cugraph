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

from cugraph_pyg.nn import HeteroGATConv as CuGraphHeteroGATConv
from cugraph.utilities.utils import import_optional, MissingModule

torch = import_optional("torch")
torch_geometric = import_optional("torch_geometric")

ATOL = 1e-6


@pytest.mark.cugraph_ops
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(
    isinstance(torch_geometric, MissingModule), reason="torch_geometric not available"
)
@pytest.mark.parametrize("heads", [1, 3, 10])
@pytest.mark.parametrize("aggr", ["sum", "mean"])
def test_hetero_gat_conv_equality(sample_pyg_hetero_data, aggr, heads):
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, GATConv

    device = torch.device("cuda:0")
    data = HeteroData(sample_pyg_hetero_data).to(device)

    in_channels_dict = {k: v.size(1) for k, v in data.x_dict.items()}
    out_channels = 2

    convs_dict = {}
    kwargs1 = dict(heads=heads, add_self_loops=False, bias=False)
    for edge_type in data.edge_types:
        src_t, _, dst_t = edge_type
        in_channels_src, in_channels_dst = data.x_dict[src_t].size(-1), data.x_dict[
            dst_t
        ].size(-1)
        if src_t == dst_t:
            convs_dict[edge_type] = GATConv(in_channels_src, out_channels, **kwargs1)
        else:
            convs_dict[edge_type] = GATConv(
                (in_channels_src, in_channels_dst), out_channels, **kwargs1
            )

    conv1 = HeteroConv(convs_dict, aggr=aggr).to(device)
    kwargs2 = dict(
        heads=heads,
        aggr=aggr,
        node_types=data.node_types,
        edge_types=data.edge_types,
        bias=False,
    )
    conv2 = CuGraphHeteroGATConv(in_channels_dict, out_channels, **kwargs2).to(device)

    # copy over linear and attention weights
    w_src, w_dst = conv2.split_tensors(conv2.lin_weights, dim=0)
    with torch.no_grad():
        for edge_type in conv2.edge_types:
            src_t, _, dst_t = edge_type
            w_src[edge_type][:, :] = conv1.convs[edge_type].lin_src.weight[:, :]
            if w_dst[edge_type] is not None:
                w_dst[edge_type][:, :] = conv1.convs[edge_type].lin_dst.weight[:, :]

            conv2.attn_weights[edge_type][: heads * out_channels] = conv1.convs[
                edge_type
            ].att_src.data.flatten()
            conv2.attn_weights[edge_type][heads * out_channels :] = conv1.convs[
                edge_type
            ].att_dst.data.flatten()

    out1 = conv1(data.x_dict, data.edge_index_dict)
    out2 = conv2(data.x_dict, data.edge_index_dict)

    for node_type in data.node_types:
        assert torch.allclose(out1[node_type], out2[node_type], atol=ATOL)

    loss1 = 0
    loss2 = 0
    for node_type in data.node_types:
        loss1 += out1[node_type].mean()
        loss2 += out2[node_type].mean()

    loss1.backward()
    loss2.backward()

    # check gradient w.r.t attention weights
    out_dim = heads * out_channels
    for edge_type in conv2.edge_types:
        assert torch.allclose(
            conv1.convs[edge_type].att_src.grad.flatten(),
            conv2.attn_weights[edge_type].grad[:out_dim],
            atol=ATOL,
        )
        assert torch.allclose(
            conv1.convs[edge_type].att_dst.grad.flatten(),
            conv2.attn_weights[edge_type].grad[out_dim:],
            atol=ATOL,
        )

    # check gradient w.r.t linear weights
    grad_lin_weights_ref = dict.fromkeys(out1.keys())
    for node_t, (rels_as_src, rels_as_dst) in conv2.relations_per_ntype.items():
        grad_list = []
        for rel_t in rels_as_src:
            grad_list.append(conv1.convs[rel_t].lin_src.weight.grad.clone())
        for rel_t in rels_as_dst:
            grad_list.append(conv1.convs[rel_t].lin_dst.weight.grad.clone())
        assert len(grad_list) > 0
        grad_lin_weights_ref[node_t] = torch.vstack(grad_list)

    for node_type in conv2.lin_weights:
        assert torch.allclose(
            grad_lin_weights_ref[node_type],
            conv2.lin_weights[node_type].grad,
            atol=ATOL,
        )
