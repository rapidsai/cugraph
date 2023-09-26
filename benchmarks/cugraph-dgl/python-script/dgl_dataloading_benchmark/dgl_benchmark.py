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


import dgl
import torch
import pandas as pd
import os
import time
import json
import random
import numpy as np
from argparse import ArgumentParser


def load_edges_from_disk(parquet_path, replication_factor, input_meta):
    """
    Load the edges from disk into a graph data dictionary.
    Args:
        parquet_path: Path to the parquet directory.
        replication_factor: Number of times to replicate the edges.
        input_meta: Input meta data.
    Returns:
        dict: Dictionary of edge types to a tuple of (src, dst)
    """
    graph_data = {}
    for edge_type in input_meta["num_edges"].keys():
        print(
            f"Loading edge index for edge type {edge_type}"
            f"for replication factor = {replication_factor}"
        )
        can_edge_type = tuple(edge_type.split("__"))
        # TODO: Rename `edge_index` to a better name
        ei = pd.read_parquet(
            os.path.join(parquet_path, edge_type, "edge_index.parquet")
        )
        ei = {
            "src": torch.from_numpy(ei.src.values),
            "dst": torch.from_numpy(ei.dst.values),
        }
        if replication_factor > 1:
            src_ls = [ei["src"]]
            dst_ls = [ei["dst"]]
            for r in range(1, replication_factor):
                new_src = ei["src"] + (
                    r * input_meta["num_nodes"][can_edge_type[0]]
                )
                src_ls.append(new_src)
                new_dst = ei["dst"] + (
                    r * input_meta["num_nodes"][can_edge_type[2]]
                )
                dst_ls.append(new_dst)

            ei["src"] = torch.cat(src_ls).contiguous()
            ei["dst"] = torch.cat(dst_ls).contiguous()
        graph_data[can_edge_type] = ei["src"], ei["dst"]
    print("Graph Data compiled")
    return graph_data


def load_node_labels(dataset_path, replication_factor, input_meta):
    num_nodes_dict = {
        node_type: t * replication_factor
        for node_type, t in input_meta["num_nodes"].items()
    }
    node_data = {}
    for node_type in input_meta["num_nodes"].keys():
        node_data[node_type] = {}
        label_path = os.path.join(
            dataset_path, "parquet", node_type, "node_label.parquet"
        )
        if os.path.exists(label_path):
            node_label = pd.read_parquet(label_path)
            if replication_factor > 1:
                base_num_nodes = input_meta["num_nodes"][node_type]
                dfr = pd.DataFrame(
                    {
                        "node": pd.concat(
                            [
                                node_label.node + (r * base_num_nodes)
                                for r in range(1, replication_factor)
                            ]
                        ),
                        "label": pd.concat(
                            [
                                node_label.label
                                for r in range(1, replication_factor)
                            ]
                        ),
                    }
                )
                node_label = pd.concat([node_label, dfr]).reset_index(
                    drop=True
                )

            node_label_tensor = torch.full(
                (num_nodes_dict[node_type],), -1, dtype=torch.float32
            )
            node_label_tensor[
                torch.as_tensor(node_label.node.values)
            ] = torch.as_tensor(node_label.label.values)

            del node_label
            node_data[node_type]["train_idx"] = (
                (node_label_tensor > -1).contiguous().nonzero().view(-1)
            )
            node_data[node_type]["y"] = node_label_tensor.contiguous()
        else:
            node_data[node_type]["num_nodes"] = num_nodes_dict[node_type]
    return node_data


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
    graph_data = load_edges_from_disk(
        parquet_path, replication_factor, input_meta
    )
    node_data = load_node_labels(dataset_path, replication_factor, input_meta)
    g = dgl.heterograph(graph_data)

    return g, node_data


def create_dataloader(g, train_idx, batch_size, fanouts, use_uva):
    """
    Create a DGL dataloader from a DGL graph.
    Args:
        g: DGLGraph to create the dataloader from.
        train_idx: Tensor containing the training indices.
        batch_size: Batch size to use for the dataloader.
        fanouts: List of fanouts to use for the dataloader.
        use_uva: Whether to use unified virtual address space.
    Returns:
        DGLGraph: DGLGraph with the loaded dataset.
    """
    
    print("Creating dataloader", flush=True)
    st = time.time()
    if use_uva:
        train_idx = {k: v.to("cuda") for k, v in train_idx.items()}
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=fanouts)
    dataloader = dgl.dataloading.DataLoader(
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
    print(f"Time to create dataloader = {et - st:.2f} seconds")
    return dataloader


def dataloading_benchmark(g, train_idx, fanouts, batch_sizes, use_uva):
    """
    Run the dataloading benchmark.
    Args:
        g: DGLGraph
        train_idx: Tensor containing the training indices.
        fanouts: List of fanouts to use for the dataloader.
        batch_sizes: List of batch sizes to use for the dataloader.
        use_uva: Whether to use unified virtual address space.
    """
    time_ls = []
    for fanout in fanouts:
        for batch_size in batch_sizes:
            dataloader = create_dataloader(
                g,
                train_idx,
                batch_size=batch_size,
                fanouts=fanout,
                use_uva=use_uva,
            )
            dataloading_st = time.time()
            for input_nodes, output_nodes, blocks in dataloader:
                pass
            dataloading_et = time.time()
            dataloading_time = dataloading_et - dataloading_st
            time_d = {
                "fanout": fanout,
                "batch_size": batch_size,
                "dataloading_time_per_epoch": dataloading_time,
                "dataloading_time_per_batch": dataloading_time / len(dataloader),
                "num_edges": g.num_edges(),
                "num_batches": len(dataloader),
            }
            time_ls.append(time_d)

            print("Dataloading completed")
            print(f"Fanout = {fanout}, batch_size = {batch_size}")
            print(
                f"Time taken {dataloading_time:.2f} ",
                f"seconds for num batches {len(dataloader)}",
                flush=True,
            )
            print("==============================================")
    return time_ls

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, default="/datasets/abarghi/ogbn_papers100M"
    )
    parser.add_argument("--replication_factors", type=str, default="1,2,4,8")
    parser.add_argument(
        "--fanouts", type=str, default="25_25,10_10_10,5_10_20"
    )
    parser.add_argument("--batch_sizes", type=str, default="512,1024")
    parser.add_argument("--do_not_use_uva", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.do_not_use_uva:
        use_uva = False
    else:
        use_uva = True
    set_seed(args.seed)
    replication_factors = [int(x) for x in args.replication_factors.split(",")]
    fanouts = [[int(y) for y in x.split("_")] for x in args.fanouts.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print("Running dgl dataloading benchmark with the following parameters:")
    print(f"Dataset path = {args.dataset_path}")
    print(f"Replication factors = {replication_factors}")
    print(f"Fanouts = {fanouts}")
    print(f"Batch sizes = {batch_sizes}")
    print(f"Use UVA = {use_uva}")
    print("==============================================")

    time_ls = []
    for replication_factor in replication_factors:
        st = time.time()
        g, node_data = create_dgl_graph_from_disk(
            dataset_path=args.dataset_path,
            replication_factor=replication_factor,
        )
        et = time.time()
        print(f"Replication factor = {replication_factor}")
        print(
            f"G has {g.num_edges()} edges and took",
            f" {et - st:.2f} seconds to load"
        )
        train_idx = {"paper": node_data["paper"]["train_idx"]}
        r_time_ls = dataloading_benchmark(
            g, train_idx, fanouts, batch_sizes, use_uva=use_uva
        )
        print(
            "Benchmark completed for replication factor = ", replication_factor
        )
        print("==============================================")
        # Add replication factor to the time list
        [
            x.update({"replication_factor": replication_factor})
            for x in r_time_ls
        ]
        time_ls.extend(r_time_ls)

    df = pd.DataFrame(time_ls)
    df.to_csv("dgl_dataloading_benchmark.csv", index=False)
    print("Benchmark completed for all replication factors")
    print("==============================================")
