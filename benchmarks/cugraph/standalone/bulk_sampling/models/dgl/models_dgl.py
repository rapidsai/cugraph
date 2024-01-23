# Copyright (c) 2018-2024, NVIDIA CORPORATION.
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
import torch.nn.functional as F


class GraphSAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        model_backend="dgl",
    ):
        if model_backend == "dgl":
            from dgl.nn import SAGEConv
        else:
            from cugraph_dgl.nn import SAGEConv

        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                SAGEConv(in_channels, hidden_channels, aggregator_type="mean")
            )
            in_channels = hidden_channels
        self.convs.append(
            SAGEConv(hidden_channels, out_channels, aggregator_type="mean")
        )

    def forward(self, blocks, x):
        for i, conv in enumerate(self.convs):
            x = conv(blocks[i], x)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5)
        return x


def create_model(feat_size, num_classes, num_layers, model_backend="dgl"):
    model = GraphSAGE(
        feat_size, 64, num_classes, num_layers, model_backend=model_backend
    )
    model = model.to("cuda")
    model.train()
    return model
