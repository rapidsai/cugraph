# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Example modified from:
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/node_classification.py

# Ignore Warning
import warnings
import time
import cugraph_dgl
import cugraph_dgl.dataloading
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    NeighborSampler,
    MultiLayerFullNeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import argparse

warnings.filterwarnings("ignore")


def set_allocators():
    import rmm
    import cudf
    import cupy
    from rmm.allocators.torch import rmm_torch_allocator
    from rmm.allocators.cupy import rmm_cupy_allocator

    mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(mr)
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
    cupy.cuda.set_allocator(rmm_cupy_allocator)
    cudf.set_option("spill", True)


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
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

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        all_node_ids = torch.arange(0, g.num_nodes()).to(device)
        feat = g.get_node_storage(key="feat", ntype="_N").fetch(
            all_node_ids, device=device
        )

        if isinstance(g, cugraph_dgl.Graph):
            sampler = cugraph_dgl.sampling.NeighborSampler(-1)
            loader_cls = cugraph_dgl.dataloading.FutureDataLoader
        else:
            sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
            loader_cls = DataLoader
        dataloader = loader_cls(
            g,
            torch.arange(g.num_nodes()).to(g.device),
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


def evaluate(model, graph, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            if isinstance(graph.ndata["feat"], dict):
                x = graph.ndata["feat"]["_N"][input_nodes]
                label = graph.ndata["label"]["_N"][output_nodes]
            else:
                x = graph.ndata["feat"][input_nodes]
                label = graph.ndata["label"][output_nodes]
            ys.append(label)
            y_hats.append(model(blocks, x))
    num_classes = y_hats[0].shape[1]
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size)  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"]
        if isinstance(label, dict):
            label = label["_N"]
        label = label[nid].to(device).to(pred.device)
        num_classes = pred.shape[1]
        return MF.accuracy(pred, label, task="multiclass", num_classes=num_classes)


def train(args, device, g, dataset, model):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)

    use_uva = args.mode == "mixed"
    batch_size = 1024
    fanouts = [5, 10, 15]
    if isinstance(g, cugraph_dgl.Graph):
        sampler = cugraph_dgl.dataloading.NeighborSampler(fanouts)
        loader_cls = cugraph_dgl.dataloading.FutureDataLoader
    else:
        sampler = NeighborSampler(fanouts)
        loader_cls = DataLoader
    train_dataloader = loader_cls(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )
    val_dataloader = loader_cls(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        st = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            if isinstance(g.ndata["feat"], dict):
                x = g.ndata["feat"]["_N"][input_nodes]
                y = g.ndata["label"]["_N"][output_nodes]
            else:
                x = g.ndata["feat"][input_nodes]
                y = g.ndata["label"][output_nodes]

            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        et = time.time()

        print(f"Time taken for epoch {epoch} with batch_size {batch_size} = {et-st} s")
        acc = evaluate(model, g, val_dataloader)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, total_loss / (it + 1), acc.item()
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="gpu_cugraph_dgl",
        choices=["cpu", "mixed", "gpu_dgl", "gpu_cugraph_dgl"],
        help="Training mode."
        " 'cpu' for CPU training,"
        " 'mixed' for CPU-GPU mixed training, "
        " 'gpu_dgl' for pure-GPU training, "
        " 'gpu_cugraph_dgl' for pure-GPU training.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    if args.mode == "gpu_cugraph_dgl":
        set_allocators()
    print(f"Training in {args.mode} mode.")

    # load and preprocess dataset
    print("Loading data")
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
    g = dataset[0]
    g = dgl.add_self_loop(g)
    if args.mode == "gpu_cugraph_dgl":
        g = cugraph_dgl.cugraph_dgl_graph_from_heterograph(g.to("cuda"))
        del dataset.g

    else:
        g = g.to("cuda" if args.mode == "gpu_dgl" else "cpu")
    device = torch.device(
        "cpu" if args.mode == "cpu" or args.mode == "mixed" else "cuda"
    )

    # create GraphSAGE model
    feat_shape = g.ndata["feat"].shape[1]
    print(feat_shape)

    in_size = feat_shape
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    # model training
    print("Training...")
    train(args, device, g, dataset, model)

    # test the model
    print("Testing...")
    acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
    print("Test Accuracy {:.4f}".format(acc.item()))
