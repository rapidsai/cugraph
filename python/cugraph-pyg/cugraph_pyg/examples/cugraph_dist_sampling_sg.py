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

# This example shows how to use cuGraph nccl-only comms, pylibcuGraph,
# and PyTorch to run a single-GPU sampling workflow.  Most users of the
# GNN packages will not interact with cuGraph directly.  This example
# is intented for users who want to extend cuGraph within a PyTorch workflow.

import os
import re
import tempfile

import numpy as np
import torch
import torch.multiprocessing as tmp
import torch.distributed as dist

import cudf

from cugraph.gnn import (
    DistSampleWriter,
    UniformNeighborSampler,
)

from pylibcugraph import SGGraph, ResourceHandle, GraphProperties

from ogb.nodeproppred import NodePropPredDataset


def sample(edgelist, directory):
    src = cudf.Series(edgelist[0])
    dst = cudf.Series(edgelist[1])

    seeds = cudf.Series(np.arange(0, 50))

    print("constructing graph")
    G = SGGraph(
        ResourceHandle(),
        GraphProperties(is_multigraph=True, is_symmetric=False),
        src,
        dst,
    )
    print("graph constructed")

    sample_writer = DistSampleWriter(directory=directory, batches_per_partition=2)
    sampler = UniformNeighborSampler(
        G,
        sample_writer,
        fanout=[5, 5],
    )

    sampler.sample_from_nodes(seeds, batch_size=16, random_state=62)


def main():
    dataset = NodePropPredDataset("ogbn-products")
    el = dataset[0][0]["edge_index"].astype("int64")

    with tempfile.TemporaryDirectory() as directory:
        sample(el, directory)

        print("Printing samples...")
        for file in os.listdir(directory):
            m = re.match(r"batch=([0-9]+)\.([0-9]+)\-([0-9]+)\.([0-9]+)\.parquet", file)
            rank, start, _, end = int(m[1]), int(m[2]), int(m[3]), int(m[4])
            print(f"File: {file} (batches {start} to {end} for rank {rank})")
            print(cudf.read_parquet(os.path.join(directory, file)))
            print("\n")


if __name__ == "__main__":
    main()
