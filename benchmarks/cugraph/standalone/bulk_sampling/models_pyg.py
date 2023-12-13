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

import torch

from torch_geometric.nn import SAGEConv
from torch_geometric.utils.trim_to_layer import TrimToLayer

import torch.nn as nn
import torch.nn.functional as F


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr="mean"))
        for _ in range(num_layers - 2):
            conv = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
            self.convs.append(conv)

        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr="mean"))

        self._trim = TrimToLayer()

    def forward(self, x, edge, num_sampled_nodes, num_sampled_edges):
        edge = edge.cuda()
        x = x.cuda().to(torch.float32)

        for i, conv in enumerate(self.convs):
            x, edge, _ = self._trim(
                i, num_sampled_nodes, num_sampled_edges, x, edge, None
            )

            s = x.shape[0]
            x = conv(x, edge, size=(s, s))
            x = F.relu(x)
            x = F.dropout(x, p=0.5)

        x = x.narrow(dim=0, start=0, length=x.shape[0] - num_sampled_nodes[1])

        # assert x.shape[0] == num_sampled_nodes[0]
        return x
