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

# A graphsage GNN model using dgl for node classification
# with three layers and mean aggregation
import time
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from cugraph_dgl.nn import SAGEConv
import tqdm


class Sage(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # 2-layer GraphSAGE-mean
        self.layers.append(SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l_id, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l_id != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, batch_size, device):
        """
        Inference with the GraphSAGE model on
        full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        batch_size : the node number of each inference output
        device : the inference device
        """
        # During inference with sampling,
        # multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.
        # The nodes on each layer are of course splitted in batches.

        all_node_ids = torch.arange(0, g.num_nodes()).to(device)
        feat = g.get_node_storage(key="feat", ntype="_N").fetch(
            all_node_ids, device=device
        )
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(
            1, prefetch_node_feats=["feat"]
        )
        dataloader = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes(), dtype=torch.int32).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l_id, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l_id != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l_id != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def layerwise_infer(graph, nid, model, batch_size, device):
    model.eval()
    with torch.no_grad():
        pred = model.module.inference(
            graph, batch_size, device
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"]
        if isinstance(label, dict):
            label = label["_N"]
        label = label[nid].to(pred.device)
        num_classes = pred.shape[1]
        label = label.squeeze(1)
        return MF.accuracy(pred, label, task="multiclass", num_classes=num_classes)


def train_model(model, g, opt, train_dataloader, num_epochs, rank, val_nid):
    g.ndata["feat"]["_N"] = g.ndata["feat"]["_N"].to("cuda")
    g.ndata["label"]["_N"] = g.ndata["label"]["_N"].to("cuda")
    st = time.time()
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for _, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = g.ndata["feat"]["_N"][input_nodes]
            y = g.ndata["label"]["_N"][output_nodes]
            y_hat = model(blocks, x)
            y = y.squeeze(1)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(
            f"total loss: {total_loss} for epoch = {epoch} for rank = {rank}",
            flush=True,
        )
    et = time.time()
    print(
        f"Total time taken for num_epochs {num_epochs} "
        f"with batch_size {train_dataloader._batch_size} = {et-st} s on rank ={rank}"
    )
    if rank == 0:
        val_acc = layerwise_infer(g, val_nid, model, 1024 * 5, "cuda")
        print("---" * 30)
        print("Validation Accuracy {:.4f}".format(val_acc))
