# Copyright (c) 2024, NVIDIA CORPORATION.
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

# This example illustrates link classification using the ogbl-wikikg2 dataset.
# The contrived task is to predict the relation type.

import argparse

from typing import Tuple, Dict, Any

import torch
import cupy

import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
from rmm.allocators.torch import rmm_torch_allocator

# Must change allocators immediately upon import
# or else other imports will cause memory to be
# allocated and prevent changing the allocator
rmm.reinitialize(devices=[0], pool_allocator=True, managed_memory=True)
cupy.cuda.set_allocator(rmm_cupy_allocator)
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

import torch.nn.functional as F  # noqa: E402
from torch.nn import Parameter  # noqa: E402
from torch_geometric.nn import FastRGCNConv, GAE  # noqa: E402
import torch_geometric  # noqa: E402
import cugraph_pyg  # noqa: E402

# Enable cudf spilling to save gpu memory
from cugraph.testing.mg_utils import enable_spilling  # noqa: E402

enable_spilling()


class RGCNEncoder(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels, num_relations, num_bases=30):
        super().__init__()
        self.node_emb = Parameter(torch.empty(num_nodes, hidden_channels))
        self.conv1 = FastRGCNConv(
            hidden_channels, hidden_channels, num_relations, num_bases=num_bases
        )
        self.conv2 = FastRGCNConv(
            hidden_channels, hidden_channels, num_relations, num_bases=num_bases
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index, edge_type):
        x = self.node_emb
        x = self.conv1(x, edge_index, edge_type).relu_()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


def load_data(
    dataset_str, dataset_root: str
) -> Tuple[
    Tuple["torch_geometric.data.FeatureStore", "torch_geometric.data.GraphStore"],
    "torch_geometric.data.FeatureStore",
    Dict[str, Dict[str, "torch.Tensor"]],
    Dict[str, Any],
]:
    from ogb.linkproppred import PygLinkPropPredDataset

    data = PygLinkPropPredDataset(dataset_str, root=dataset_root)
    dataset = data[0]

    splits = data.get_edge_split()

    from cugraph_pyg.data import GraphStore, TensorDictFeatureStore

    graph_store = GraphStore()
    feature_store = TensorDictFeatureStore()
    edge_feature_store = TensorDictFeatureStore()
    meta = {}

    graph_store[("n", "e", "n"), "coo"] = dataset.edge_index
    edge_feature_store[("n", "e", "n"), "rel"] = dataset.edge_reltype.pin_memory()
    meta["num_nodes"] = dataset.num_nodes
    meta["num_rels"] = dataset.edge_reltype.max() + 1

    return (feature_store, graph_store), edge_feature_store, splits, meta


def train(epoch, model, optimizer, train_loader, edge_feature_store):
    model.train()
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        r = edge_feature_store[("n", "e", "n"), "rel"][batch.e_id].flatten().cuda()
        z = model.encode(batch.edge_index, r)

        loss = model.recon_loss(z, batch.edge_index)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch: {epoch:02d}, Iteration: {i:02d}, Loss: {loss:.4f}")
            if i == 100:
                break


def test(stage, epoch, model, loader, num_steps=None):
    # TODO support ROC-AUC metric
    # Predict probabilities of future edges
    model.eval()

    rr = 0.0
    for i, (h, h_neg, t, t_neg, r) in enumerate(loader):
        if num_steps and i >= num_steps:
            break

        ei = torch.concatenate(
            [
                torch.stack([h, t]).cuda(),
                torch.stack([h_neg.flatten(), t_neg.flatten()]).cuda(),
            ],
            dim=-1,
        )

        r = torch.concatenate([r, torch.repeat_interleave(r, h_neg.shape[-1])]).cuda()

        z = model.encode(ei, r)
        q = model.decode(z, ei)

        _, ix = torch.sort(q, descending=True)
        rr += 1.0 / (1.0 + ix[0])

    print(f"epoch {epoch:02d} {stage} mrr:", rr / i)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--num_neg", type=int, default=500)
    parser.add_argument("--fan_out", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="ogbl-wikikg2")
    parser.add_argument("--dataset_root", type=str, default="dataset")
    parser.add_argument("--seeds_per_call", type=int, default=-1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data, edge_feature_store, splits, meta = load_data(args.dataset, args.dataset_root)

    model = GAE(
        RGCNEncoder(
            meta["num_nodes"],
            hidden_channels=args.hidden_channels,
            num_relations=meta["num_rels"],
        )
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = cugraph_pyg.loader.LinkNeighborLoader(
        data,
        [args.fan_out] * args.num_layers,
        edge_label_index=torch.stack(
            [splits["train"]["head"], splits["train"]["tail"]]
        ),
        local_seeds_per_call=args.seeds_per_call if args.seeds_per_call > 0 else None,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    def get_eval_loader(stage: str):
        head = splits[stage]["head"]
        tail = splits[stage]["tail"]

        head_neg = splits[stage]["head_neg"][:, : args.num_neg]
        tail_neg = splits[stage]["tail_neg"][:, : args.num_neg]

        rel = splits[stage]["relation"]

        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                head.pin_memory(),
                head_neg.pin_memory(),
                tail.pin_memory(),
                tail_neg.pin_memory(),
                rel.pin_memory(),
            ),
            batch_size=1,
            shuffle=False,
            drop_last=True,
        )

    test_loader = get_eval_loader("test")
    valid_loader = get_eval_loader("valid")

    for epoch in range(1, 1 + args.epochs):
        train(epoch, model, optimizer, train_loader, edge_feature_store)
        test("validation", epoch, model, valid_loader, num_steps=1024)

    test("test", epoch, model, test_loader, num_steps=1024)
