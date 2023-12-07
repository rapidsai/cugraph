# Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

import json
import os
import time
import dgl
from dgl.dataloading import MultiLayerNeighborSampler, DataLoader
import pandas as pd
import torch
from model import run_1_epoch
from argparse import ArgumentParser
from load_graph_feats import load_edges_from_disk, load_node_labels, load_node_features


class DataLoaderArgs:
    def __init__(self, args):
        self.dataset_path = args.dataset_path
        self.replication_factors = [int(x) for x in args.replication_factors.split(",")]
        self.fanouts = [[int(y) for y in x.split("_")] for x in args.fanouts.split(",")]
        self.batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
        self.use_uva = not args.do_not_use_uva


def create_dataloader(g, train_idx, batch_size, fanouts, use_uva):
    print("Creating dataloader", flush=True)
    st = time.time()
    if use_uva:
        train_idx = {k: v.to("cuda") for k, v in train_idx.items()}
    sampler = MultiLayerNeighborSampler(fanouts=fanouts)
    dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        num_workers=0,
        batch_size=batch_size,
        use_uva=use_uva,
        shuffle=False,
        drop_last=False,
    )
    et = time.time()
    print(f"Time to create dataloader = {et - st:.2f} seconds", flush=True)
    return dataloader


def create_dgl_graph_from_disk(dataset_path, replication_factor=1):
    """
    Create a DGL graph from a dataset on disk.
    Args:
        dataset_path: Path to the dataset on disk.
        replication_factor: Number of times to replicate the edges.
    Returns:
        DGLGraph: DGLGraph with the loaded dataset.
    """
    with open(os.path.join(dataset_path, "meta.json"), "r") as f:
        input_meta = json.load(f)

    parquet_path = os.path.join(dataset_path, "parquet")
    graph_data = load_edges_from_disk(parquet_path, replication_factor, input_meta)
    label_data = load_node_labels(dataset_path, replication_factor, input_meta)
    if replication_factor < 8:
        feat_data = load_node_features(
            dataset_path, replication_factor, node_type="paper"
        )
    else:
        feat_data = None
    print("labels and features loaded ", flush=True)

    g = dgl.heterograph(graph_data)

    return g, label_data, feat_data


def main(args):
    print(
        f"Running dgl dataloading benchmark with the following parameters:\n"
        f"Dataset path = {args.dataset_path}\n"
        f"Replication factors = {args.replication_factors}\n"
        f"Fanouts = {args.fanouts}\n"
        f"Batch sizes = {args.batch_sizes}\n"
        f"Use UVA = {args.use_uva}\n"
        f"{'=' * 30}"
    )

    time_ls = []
    for replication_factor in args.replication_factors:
        start_time = time.time()
        g, label_data, feat_data = create_dgl_graph_from_disk(
            args.dataset_path, replication_factor
        )
        elapsed_time = time.time() - start_time

        print(
            f"Replication factor = {replication_factor}\n"
            f"G has {g.num_edges():,} edges and took {elapsed_time:.2f} seconds to load",
            flush=True,
        )

        train_idx = {"paper": label_data["paper"]["train_idx"]}
        y = label_data["paper"]["y"]
        r_time_ls = e2e_benchmark(
            g,
            feat_data,
            y,
            train_idx,
            args.fanouts,
            args.batch_sizes,
            use_uva=args.use_uva,
        )
        [x.update({"replication_factor": replication_factor}) for x in r_time_ls]
        [x.update({"num_edges": g.num_edges()}) for x in r_time_ls]
        time_ls.extend(r_time_ls)

        print(
            f"Benchmark completed for replication factor = {replication_factor}\n{'=' * 30}",
            flush=True,
        )

    df = pd.DataFrame(time_ls)
    df.to_csv("dgl_e2e_benchmark.csv", index=False)
    print(f"Benchmark completed for all replication factors\n{'=' * 30}", flush=True)


def e2e_benchmark(g, feat, y, train_idx, fanouts, batch_sizes, use_uva):
    """
    Run the e2e_benchmark
    Args:
        g: DGLGraph
        feat: Tensor containing the features.
        y: Tensor containing the labels.
        train_idx: Tensor containing the training indices.
        fanouts: List of fanouts to use for the dataloader.
        batch_sizes: List of batch sizes to use for the dataloader.
        use_uva: Whether to use unified virtual address space.
        model_backend: Backend of model to use.
    """
    time_ls = []
    for fanout in fanouts:
        for batch_size in batch_sizes:
            dataloader = create_dataloader(g, train_idx, batch_size, fanout, use_uva)
            time_d = run_1_epoch(
                dataloader, feat, y, fanout, batch_size, model_backend="dgl"
            )
            time_ls.append(time_d)
            print("=" * 30)
    return time_ls


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, default="/raid/vjawa/ogbn_papers100M"
    )
    parser.add_argument("--replication_factors", type=str, default="2")
    parser.add_argument("--fanouts", type=str, default="10_10_10")
    parser.add_argument("--batch_sizes", type=str, default="512,1024,8192,16384")
    parser.add_argument("--do_not_use_uva", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    main(DataLoaderArgs(arguments))
