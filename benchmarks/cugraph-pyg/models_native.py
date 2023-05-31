import torch

from torch_geometric.nn import SAGEConv
from torch_geometric.utils.trim_to_layer import TrimToLayer

import torch.nn as nn
import torch.nn.functional as F

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))
        for _ in range(num_layers - 2):
            conv = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
            self.convs.append(conv)
        
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))

        self._trim = TrimToLayer()

    def forward(self, x, edge, num_sampled_nodes, num_sampled_edges):
        for i, conv in enumerate(self.convs):
            edge = edge.cuda()
            x = x.cuda().to(torch.float32)

            x, edge, _ = self._trim(
                i,
                num_sampled_nodes,
                num_sampled_edges,
                x,
                edge,
                None
            )

            s = x.shape[0]
            x = conv(x, edge, size=(s, s))
            x = F.relu(x)
            x = F.dropout(x, p=0.5)

        return x
