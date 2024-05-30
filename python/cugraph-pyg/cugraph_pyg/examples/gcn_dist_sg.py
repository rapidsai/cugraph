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

import time
import argparse
import tempfile
import os

from typing import Optional, Tuple, Dict

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
import torch_geometric  # noqa: E402
import cugraph_pyg  # noqa: E402
from cugraph_pyg.loader import NeighborLoader  # noqa: E402

# Enable cudf spilling to save gpu memory
from cugraph.testing.mg_utils import enable_spilling  # noqa: E402

enable_spilling()


def train(epoch: int):
    model.train()
    for i, batch in enumerate(train_loader):
        if i == warmup_steps:
            torch.cuda.synchronize()
            start_avg_time = time.perf_counter()
        batch = batch.to(device)

        optimizer.zero_grad()
        batch_size = batch.batch_size
        out = model(batch.x, batch.edge_index)[:batch_size]
        y = batch.y[:batch_size].view(-1).to(torch.long)

        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch: {epoch:02d}, Iteration: {i}, Loss: {loss:.4f}")
    torch.cuda.synchronize()
    print(
        f"Average Training Iteration Time (s/iter): \
            {(time.perf_counter() - start_avg_time)/(i-warmup_steps):.6f}"
    )


@torch.no_grad()
def test(loader: NeighborLoader, val_steps: Optional[int] = None):
    model.eval()

    total_correct = total_examples = 0
    for i, batch in enumerate(loader):
        if val_steps is not None and i >= val_steps:
            break
        batch = batch.to(device)
        batch_size = batch.batch_size
        out = model(batch.x, batch.edge_index)[:batch_size]
        pred = out.argmax(dim=-1)
        y = batch.y[:batch_size].view(-1).to(torch.long)

        total_correct += int((pred == y).sum())
        total_examples += y.size(0)

    return total_correct / total_examples


def create_loader(
    data, num_neighbors, input_nodes, replace, batch_size, samples_dir, stage_name
):
    directory = os.path.join(samples_dir, stage_name)
    os.mkdir(directory)
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=input_nodes,
        replace=replace,
        batch_size=batch_size,
        directory=directory,
    )


def load_data(
    dataset, dataset_root
) -> Tuple[
    Tuple[torch_geometric.data.FeatureStore, torch_geometric.data.GraphStore],
    Dict[str, torch.Tensor],
    int,
    int,
]:
    from ogb.nodeproppred import PygNodePropPredDataset

    dataset = PygNodePropPredDataset(dataset, root=dataset_root)
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    graph_store = cugraph_pyg.data.GraphStore()
    graph_store[
        ("node", "rel", "node"), "coo", False, (data.num_nodes, data.num_nodes)
    ] = data.edge_index

    feature_store = cugraph_pyg.data.TensorDictFeatureStore()
    feature_store["node", "x"] = data.x
    feature_store["node", "y"] = data.y

    return (
        (feature_store, graph_store),
        split_idx,
        dataset.num_features,
        dataset.num_classes,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--fan_out", type=int, default=30)
    parser.add_argument("--tempdir_root", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default="dataset")
    parser.add_argument("--dataset", type=str, default="ogbn-products")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    wall_clock_start = time.perf_counter()
    device = torch.device("cuda")

    data, split_idx, num_features, num_classes = load_data(
        args.dataset, args.dataset_root
    )

    with tempfile.TemporaryDirectory(dir=args.tempdir_root) as samples_dir:
        loader_kwargs = {
            "data": data,
            "num_neighbors": [args.fan_out] * args.num_layers,
            "replace": False,
            "batch_size": args.batch_size,
            "samples_dir": samples_dir,
        }

        train_loader = create_loader(
            input_nodes=split_idx["train"],
            stage_name="train",
            **loader_kwargs,
        )

        val_loader = create_loader(
            input_nodes=split_idx["valid"],
            stage_name="val",
            **loader_kwargs,
        )

        test_loader = create_loader(
            input_nodes=split_idx["test"],
            stage_name="test",
            **loader_kwargs,
        )

        model = torch_geometric.nn.models.GCN(
            num_features,
            args.hidden_channels,
            args.num_layers,
            num_classes,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=0.0005
        )

        warmup_steps = 20

        torch.cuda.synchronize()
        prep_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total time before training begins (prep_time)=", prep_time, "seconds")
        print("Beginning training...")
        for epoch in range(1, 1 + args.epochs):
            train(epoch)
            val_acc = test(val_loader, val_steps=100)
            print(f"Val Acc: ~{val_acc:.4f}")

        test_acc = test(test_loader)
        print(f"Test Acc: {test_acc:.4f}")
        total_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total Program Runtime (total_time) =", total_time, "seconds")
        print("total_time - prep_time =", total_time - prep_time, "seconds")
