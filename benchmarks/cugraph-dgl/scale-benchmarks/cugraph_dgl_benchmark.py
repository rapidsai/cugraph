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
import os

os.environ["LIBCUDF_CUFILE_POLICY"] = "KVIKIO"
os.environ["KVIKIO_NTHREADS"] = "64"
os.environ["RAPIDS_NO_INITIALIZE"] = "1"
import json
import pandas as pd
import os
import time
from rmm.allocators.torch import rmm_torch_allocator
import rmm
import torch
from cugraph_dgl.dataloading import HomogenousBulkSamplerDataset
from model import run_1_epoch
from argparse import ArgumentParser
from load_graph_feats import load_node_labels, load_node_features


def create_dataloader(sampled_dir, total_num_nodes, sparse_format, return_type):
    print("Creating dataloader", flush=True)
    st = time.time()
    dataset = HomogenousBulkSamplerDataset(
        total_num_nodes,
        edge_dir="in",
        sparse_format=sparse_format,
        return_type=return_type,
    )

    dataset.set_input_files(sampled_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, collate_fn=lambda x: x, shuffle=False, num_workers=0, batch_size=None
    )
    et = time.time()
    print(f"Time to create dataloader = {et - st:.2f} seconds", flush=True)
    return dataloader


def setup_common_pool():
    rmm.reinitialize(initial_pool_size=5e9, pool_allocator=True)
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)


def main(args):
    print(
        f"Running cugraph-dgl dataloading benchmark with the following parameters:\n"
        f"Dataset path = {args.dataset_path}\n"
        f"Sampling path = {args.sampling_path}\n"
    )
    with open(os.path.join(args.dataset_path, "meta.json"), "r") as f:
        input_meta = json.load(f)

    sampled_dirs = [
        os.path.join(args.sampling_path, f) for f in os.listdir(args.sampling_path)
    ]

    time_ls = []
    for sampled_dir in sampled_dirs:
        with open(os.path.join(sampled_dir, "output_meta.json"), "r") as f:
            sampled_meta_d = json.load(f)

        replication_factor = sampled_meta_d["replication_factor"]
        feat_load_st = time.time()
        label_data = load_node_labels(
            args.dataset_path, replication_factor, input_meta
        )["paper"]["y"]
        feat_data = feat_data = load_node_features(
            args.dataset_path, replication_factor, node_type="paper"
        )
        print(
            f"Feature and label data loading took = {time.time()-feat_load_st}",
            flush=True,
        )

        r_time_ls = e2e_benchmark(sampled_dir, feat_data, label_data, sampled_meta_d)
        [x.update({"replication_factor": replication_factor}) for x in r_time_ls]
        [x.update({"num_edges": sampled_meta_d["total_num_edges"]}) for x in r_time_ls]
        time_ls.extend(r_time_ls)

        print(
            f"Benchmark completed for replication factor = {replication_factor}\n{'=' * 30}",
            flush=True,
        )

    df = pd.DataFrame(time_ls)
    df.to_csv("cugraph_dgl_e2e_benchmark.csv", index=False)
    print(f"Benchmark completed for all replication factors\n{'=' * 30}", flush=True)


def e2e_benchmark(
    sampled_dir: str, feat: torch.Tensor, y: torch.Tensor, sampled_meta_d: dict
):
    """
    Run the e2e_benchmark
    Args:
        sampled_dir: directory containing the sampled graph
        feat: node features
        y: node labels
        sampled_meta_d: dictionary containing the sampled graph metadata
    """
    time_ls = []

    # TODO: Make this a parameter in bulk sampling script
    sampled_meta_d["sparse_format"] = "csc"
    sampled_dir = os.path.join(sampled_dir, "samples")
    dataloader = create_dataloader(
        sampled_dir,
        sampled_meta_d["total_num_nodes"],
        sampled_meta_d["sparse_format"],
        return_type="cugraph_dgl.nn.SparseGraph",
    )
    time_d = run_1_epoch(
        dataloader,
        feat,
        y,
        fanout=sampled_meta_d["fanout"],
        batch_size=sampled_meta_d["batch_size"],
        model_backend="cugraph_dgl",
    )
    time_ls.append(time_d)
    print("=" * 30)
    return time_ls


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, default="/raid/vjawa/ogbn_papers100M/"
    )
    parser.add_argument(
        "--sampling_path",
        type=str,
        default="/raid/vjawa/nov_1_bulksampling_benchmarks/",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_common_pool()
    arguments = parse_arguments()
    main(arguments)
