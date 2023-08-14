import torch

from torch_geometric.nn import CuGraphSAGEConv
from torch_geometric.utils.trim_to_layer import TrimToLayer

import torch.nn as nn
import torch.nn.functional as F

class CuGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(CuGraphSAGEConv(in_channels, hidden_channels, aggr='mean'))
        for _ in range(num_layers - 2):
            conv = CuGraphSAGEConv(hidden_channels, hidden_channels, aggr='mean')
            self.convs.append(conv)
        
        self.convs.append(CuGraphSAGEConv(hidden_channels, out_channels, aggr='mean'))

        self._trim = TrimToLayer()

    def forward(self, x, edge, num_sampled_nodes, num_sampled_edges):
        s = x.shape[0]
        edge = list(CuGraphSAGEConv.to_csc(edge.cuda(), (s, s)))
        x = x.cuda().to(torch.float32)

        for i, conv in enumerate(self.convs):
            if i > 0:
                x = x.narrow(
                    dim=0,
                    start=0,
                    length=x.size(0) - num_sampled_nodes[-i],
                )
                
                edge[0] = edge[0].narrow(
                    dim=0,
                    start=0,
                    length=edge[0].size(0) - num_sampled_edges[-i],
                )
                edge[1] = edge[1].narrow(
                    dim=0,
                    start=0,
                    length=edge[1].size(0) - num_sampled_nodes[-i]
                )
                edge[2] = x.size(0)

            x = conv(x, edge)
            x = F.relu(x)
            x = F.dropout(x, p=0.5)

        x = x.narrow(
            dim=0,
            start=0,
            length=x.shape[0] - num_sampled_nodes[1]
        )

        # assert x.shape[0] == num_sampled_nodes[0]
        return x

