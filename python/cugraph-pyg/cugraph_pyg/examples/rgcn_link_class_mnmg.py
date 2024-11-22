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

import os
import json
import argparse
import warnings

import torch

import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import FastRGCNConv, GAE
from torch.nn.parallel import DistributedDataParallel

from ogb.linkproppred import PygLinkPropPredDataset

import cugraph_pyg

from cugraph.gnn import (
    cugraph_comms_init,
    cugraph_comms_create_unique_id,
    cugraph_comms_shutdown,
)

from pylibwholegraph.torch.initialize import (
    init as wm_init,
    finalize as wm_finalize,
)


# Enable cudf spilling to save gpu memory
from cugraph.testing.mg_utils import enable_spilling

# Ensures that a CUDA context is not created on import of rapids.
# Allows pytorch to create the context instead
os.environ["RAPIDS_NO_INITIALIZE"] = "1"


def init_pytorch_worker(global_rank, local_rank, world_size, uid):
    import rmm

    rmm.reinitialize(devices=[local_rank], pool_allocator=True, managed_memory=True)

    import cupy
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)

    cugraph_comms_init(
        global_rank,
        world_size,
        uid,
        local_rank,
    )

    wm_init(global_rank, world_size, local_rank, torch.cuda.device_count())

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


def train(epoch, model, optimizer, train_loader, edge_feature_store, num_steps=None):
    model.train()
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        r = edge_feature_store[("n", "e", "n"), "rel"][batch.e_id].flatten().cuda()
        z = model.encode(batch.edge_index, r)

        loss = model.recon_loss(z, batch.edge_index)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(
                f"Epoch: {epoch:02d}, Iteration: {i:02d}, Loss: {loss:.4f}", flush=True
            )
        if num_steps and i == num_steps:
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

    print(f"epoch {epoch:02d} {stage} mrr:", rr / i, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--num_neg", type=int, default=500)
    parser.add_argument("--num_pos", type=int, default=-1)
    parser.add_argument("--fan_out", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="ogbl-wikikg2")
    parser.add_argument("--dataset_root", type=str, default="dataset")
    parser.add_argument("--seeds_per_call", type=int, default=-1)
    parser.add_argument("--n_devices", type=int, default=-1)
    parser.add_argument("--skip_partition", action="store_true")

    return parser.parse_args()


def run_train(rank, world_size, model, data, edge_feature_store, meta, splits, args):
    model = model.to(rank)
    model = GAE(DistributedDataParallel(model, device_ids=[rank]))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    eli = torch.stack([splits["train"]["head"], splits["train"]["tail"]])

    train_loader = cugraph_pyg.loader.LinkNeighborLoader(
        data,
        [args.fan_out] * args.num_layers,
        edge_label_index=eli,
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

    num_train_steps = (args.num_pos // args.batch_size) if args.num_pos > 0 else 100

    for epoch in range(1, 1 + args.epochs):
        train(
            epoch,
            model,
            optimizer,
            train_loader,
            edge_feature_store,
            num_steps=num_train_steps,
        )
        test("validation", epoch, model, valid_loader, num_steps=1024)

    test("test", epoch, model, test_loader, num_steps=1024)

    wm_finalize()
    cugraph_comms_shutdown()


def partition_data(
    data, splits, meta, edge_path, rel_path, pos_path, neg_path, meta_path
):
    # Split and save edge index
    os.makedirs(
        edge_path,
        exist_ok=True,
    )
    for (r, e) in enumerate(torch.tensor_split(data.edge_index, world_size, dim=1)):
        rank_path = os.path.join(edge_path, f"rank={r}.pt")
        torch.save(
            e.clone(),
            rank_path,
        )

    # Split and save edge reltypes
    os.makedirs(
        rel_path,
        exist_ok=True,
    )
    for (r, f) in enumerate(torch.tensor_split(data.edge_reltype, world_size)):
        rank_path = os.path.join(rel_path, f"rank={r}.pt")
        torch.save(
            f.clone(),
            rank_path,
        )

    # Split and save positive edges
    os.makedirs(
        pos_path,
        exist_ok=True,
    )
    for stage in ["train", "test", "valid"]:
        for (r, n) in enumerate(
            torch.tensor_split(
                torch.stack([splits[stage]["head"], splits[stage]["tail"]]),
                world_size,
                dim=-1,
            )
        ):
            rank_path = os.path.join(pos_path, f"rank={r}_{stage}.pt")
            torch.save(
                n.clone(),
                rank_path,
            )

    # Split and save negative edges
    os.makedirs(
        neg_path,
        exist_ok=True,
    )
    for stage in ["test", "valid"]:
        for (r, n) in enumerate(
            torch.tensor_split(
                torch.stack([splits[stage]["head_neg"], splits[stage]["tail_neg"]]),
                world_size,
                dim=1,
            )
        ):
            rank_path = os.path.join(neg_path, f"rank={r}_{stage}.pt")
            torch.save(n.clone(), rank_path)
        for (r, n) in enumerate(
            torch.tensor_split(splits[stage]["relation"], world_size, dim=-1)
        ):
            print(n)
            rank_path = os.path.join(neg_path, f"rank={r}_{stage}_relation.pt")
            torch.save(n.clone(), rank_path)

    with open(meta_path, "w") as f:
        json.dump(meta, f)


def load_partitioned_data(rank, edge_path, rel_path, pos_path, neg_path, meta_path):
    from cugraph_pyg.data import GraphStore, WholeFeatureStore, TensorDictFeatureStore

    graph_store = GraphStore()
    feature_store = TensorDictFeatureStore()
    edge_feature_store = WholeFeatureStore()

    # Load edge index
    graph_store[("n", "e", "n"), "coo"] = torch.load(
        os.path.join(edge_path, f"rank={rank}.pt")
    )

    # Load edge rel type
    edge_feature_store[("n", "e", "n"), "rel"] = torch.load(
        os.path.join(rel_path, f"rank={rank}.pt")
    )

    splits = {}

    # Load positive edges
    for stage in ["train", "test", "valid"]:
        head, tail = torch.load(os.path.join(pos_path, f"rank={rank}_{stage}.pt"))
        splits[stage] = {
            "head": head,
            "tail": tail,
        }

    # Load negative edges
    for stage in ["test", "valid"]:
        head_neg, tail_neg = torch.load(
            os.path.join(neg_path, f"rank={rank}_{stage}.pt")
        )
        relation = torch.load(
            os.path.join(neg_path, f"rank={rank}_{stage}_relation.pt")
        )
        splits[stage]["head_neg"] = head_neg
        splits[stage]["tail_neg"] = tail_neg
        splits[stage]["relation"] = relation

    with open(meta_path, "r") as f:
        meta = json.load(f)

    return (feature_store, graph_store), edge_feature_store, splits, meta


if __name__ == "__main__":
    args = parse_args()

    if "LOCAL_RANK" in os.environ:
        torch.distributed.init_process_group("nccl")
        world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(local_rank)

        # Create the uid needed for cuGraph comms
        if global_rank == 0:
            cugraph_id = [cugraph_comms_create_unique_id()]
        else:
            cugraph_id = [None]
        torch.distributed.broadcast_object_list(cugraph_id, src=0, device=device)
        cugraph_id = cugraph_id[0]

        init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id)

        # Split the data
        edge_path = os.path.join(args.dataset_root, args.dataset + "_eix_part")
        rel_path = os.path.join(args.dataset_root, args.dataset + "_rel_part")
        pos_path = os.path.join(args.dataset_root, args.dataset + "_e_pos_part")
        neg_path = os.path.join(args.dataset_root, args.dataset + "_e_neg_part")
        meta_path = os.path.join(args.dataset_root, args.dataset + "_meta.json")

        if not args.skip_partition and global_rank == 0:
            data = PygLinkPropPredDataset(args.dataset, root=args.dataset_root)
            dataset = data[0]

            splits = data.get_edge_split()

            meta = {}
            meta["num_nodes"] = int(dataset.num_nodes)
            meta["num_rels"] = int(dataset.edge_reltype.max()) + 1

            partition_data(
                dataset,
                splits,
                meta,
                edge_path=edge_path,
                rel_path=rel_path,
                pos_path=pos_path,
                neg_path=neg_path,
                meta_path=meta_path,
            )
            del data
            del dataset
            del splits
        torch.distributed.barrier()

        # Load partitions
        data, edge_feature_store, splits, meta = load_partitioned_data(
            rank=global_rank,
            edge_path=edge_path,
            rel_path=rel_path,
            pos_path=pos_path,
            neg_path=neg_path,
            meta_path=meta_path,
        )
        torch.distributed.barrier()

        model = RGCNEncoder(
            meta["num_nodes"],
            hidden_channels=args.hidden_channels,
            num_relations=meta["num_rels"],
        )

        run_train(
            global_rank, world_size, model, data, edge_feature_store, meta, splits, args
        )
    else:
        warnings.warn("This script should be run with 'torchrun`.  Exiting.")
