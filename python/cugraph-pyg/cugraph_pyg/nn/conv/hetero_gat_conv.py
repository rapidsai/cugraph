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

from typing import Optional
from collections import defaultdict

from cugraph.utilities.utils import import_optional
from pylibcugraphops.pytorch.operators import mha_gat_n2n

from .base import BaseConv

torch = import_optional("torch")
nn = import_optional("torch.nn")
torch_geometric = import_optional("torch_geometric")


class HeteroGATConv(BaseConv):
    r"""Heterogeneous graph."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        node_types: list[str],
        edge_types: list[tuple[str, str, str]],
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        bias: bool = True,
        aggr: str = "sum",
    ):
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types

        self.num_heads = heads
        self.concat_heads = concat
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.aggr = aggr

        edge_types_str = ["__".join(etype) for etype in self.edge_types]

        self.relations_per_ntype = defaultdict(lambda: ([], []))

        lin_weights = dict.fromkeys(self.node_types)

        attn_weights = dict.fromkeys(edge_types_str)

        biases = dict.fromkeys(edge_types_str)

        for edge_type in self.edge_types:
            src_type, _, dst_type = edge_type
            etype_str = "__".join(edge_type)
            self.relations_per_ntype[src_type][0].append(etype_str)
            if src_type != dst_type:
                self.relations_per_ntype[dst_type][1].append(etype_str)

            attn_weights[etype_str] = torch.empty(
                2 * self.num_heads * self.out_channels
            )

            if bias and concat:
                biases[etype_str] = torch.empty(self.num_heads * out_channels)
            elif bias:
                biases[etype_str] = torch.empty(out_channels)
            else:
                biases[etype_str] = None

        for ntype in self.node_types:
            n_src_rel = len(self.relations_per_ntype[ntype][0])
            n_dst_rel = len(self.relations_per_ntype[ntype][1])
            n_rel = n_src_rel + n_dst_rel

            lin_weights[ntype] = torch.empty(
                (n_rel * self.num_heads * self.out_channels, self.in_channels)
            )

        self.lin_weights = nn.ParameterDict(lin_weights)
        self.attn_weights = nn.ParameterDict(attn_weights)

        if bias:
            self.bias = nn.ParameterDict(biases)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def split_tensors(self, x_dict: torch.Tensor, dim: int):
        x_src_dict = {"__".join(etype): None for etype in self.edge_types}
        x_dst_dict = {"__".join(etype): None for etype in self.edge_types}

        for ntype, t in x_dict.items():
            n_src_rel = len(self.relations_per_ntype[ntype][0])
            n_dst_rel = len(self.relations_per_ntype[ntype][1])
            n_rel = n_src_rel + n_dst_rel
            t_list = torch.chunk(t, chunks=n_rel, dim=dim)

            for i, src_rel in enumerate(self.relations_per_ntype[ntype][0]):
                x_src_dict[src_rel] = t_list[i]

            for i, dst_rel in enumerate(self.relations_per_ntype[ntype][1]):
                src_type, _, dst_type = dst_rel.split("__")
                if src_type != dst_type:
                    x_dst_dict[dst_rel] = t_list[i + n_src_rel]

        return x_src_dict, x_dst_dict

    def reset_parameters(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)

        w_src, w_dst = self.split_tensors(self.lin_weights, dim=0)

        for i, edge_type in enumerate(self.edge_types):
            src_type, etype, dst_type = edge_type
            etype_str = "__".join(edge_type)
            # lin_src
            torch_geometric.nn.inits.glorot(w_src[etype_str])

            # lin_dst
            if src_type != dst_type:
                torch_geometric.nn.inits.glorot(w_dst[etype_str])

            # attn_weights
            torch_geometric.nn.inits.glorot(
                self.attn_weights[etype_str].view(-1, self.num_heads, self.out_channels)
            )

            # bias
            if self.bias is not None:
                torch_geometric.nn.inits.zeros(self.bias[etype_str])

    def forward(self, x_dict: dict, edge_index_dict: dict):
        feat_dict = {ntype: None for ntype in x_dict.keys()}

        for ntype, x in x_dict.items():
            feat_dict[ntype] = x @ self.lin_weights[ntype].T

        x_src_dict, x_dst_dict = self.split_tensors(feat_dict, dim=1)

        out_dict = defaultdict(list)

        for edge_type, edge_index in edge_index_dict.items():
            src_type, etype, dst_type = edge_type
            etype_str = "__".join(edge_type)

            csc = BaseConv.to_csc(
                edge_index, (x_dict[src_type].size(0), x_dict[dst_type].size(0))
            )

            if src_type == dst_type:
                graph = self.get_cugraph(
                    csc,
                    bipartite=False,
                )
                out = mha_gat_n2n(
                    x_src_dict[etype_str],
                    self.attn_weights[etype_str],
                    graph,
                    num_heads=self.num_heads,
                    activation="LeakyReLU",
                    negative_slope=self.negative_slope,
                    concat_heads=self.concat_heads,
                )

            else:
                graph = self.get_cugraph(
                    csc,
                    bipartite=True,
                )
                out = mha_gat_n2n(
                    (x_src_dict[etype_str], x_dst_dict[etype_str]),
                    self.attn_weights[etype_str],
                    graph,
                    num_heads=self.num_heads,
                    activation="LeakyReLU",
                    negative_slope=self.negative_slope,
                    concat_heads=self.concat_heads,
                )

            if self.bias is not None:
                out = out + self.bias[etype_str]

            out_dict[dst_type].append(out)

        for key, value in out_dict.items():
            out_dict[key] = torch_geometric.nn.conv.hetero_conv.group(value, self.aggr)

        return out_dict
