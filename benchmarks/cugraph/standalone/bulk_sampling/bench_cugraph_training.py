# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
os.environ["CUDF_SPILL"] = "1"
os.environ["LIBCUDF_CUFILE_POLICY"] = "KVIKIO"
os.environ["KVIKIO_NTHREADS"] = "8"

import argparse
import json
import warnings

import torch
import numpy as np
import pandas

import torch.distributed as dist

from datasets import OGBNPapers100MDataset

from cugraph.testing.mg_utils import enable_spilling


def init_pytorch_worker(rank: int, use_rmm_torch_allocator: bool = False) -> None:
    import cupy
    import rmm
    from pynvml.smi import nvidia_smi

    smi = nvidia_smi.getInstance()
    pool_size = 16e9  # FIXME calculate this

    rmm.reinitialize(
        devices=[rank],
        pool_allocator=True,
        initial_pool_size=pool_size,
    )

    if use_rmm_torch_allocator:
        warnings.warn(
            "Using the rmm pytorch allocator is currently unsupported."
            " The default allocator will be used instead."
        )
        # FIXME somehow get the pytorch allocator to work
        # from rmm.allocators.torch import rmm_torch_allocator
        # torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)

    cupy.cuda.Device(rank).use()
    torch.cuda.set_device(rank)

    # Pytorch training worker initialization
    torch.distributed.init_process_group(backend="nccl")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=8,
        help="# GPUs per node",
        required=False,
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
        required=False,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size",
        required=False,
    )

    parser.add_argument(
        "--fanout",
        type=str,
        default="10_10_10",
        help="Fanout",
        required=False,
    )

    parser.add_argument(
        "--sample_dir",
        type=str,
        help="Directory with stored bulk samples (required for cuGraph run)",
        required=False,
    )

    parser.add_argument(
        "--output_file",
        type=str,
        help="File to store results",
        required=True,
    )

    parser.add_argument(
        "--framework",
        type=str,
        help="The framework to test (PyG, cuGraphPyG)",
        required=True,
    )

    parser.add_argument(
        "--use_wholegraph",
        action="store_true",
        help="Whether to use WholeGraph feature storage",
        required=False,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="GraphSAGE",
        help="The model to use (currently only GraphSAGE supported)",
        required=False,
    )

    parser.add_argument(
        "--replication_factor",
        type=int,
        default=1,
        help="The replication factor for the dataset",
        required=False,
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="The directory where datasets are stored",
        required=True,
    )

    parser.add_argument(
        "--train_split",
        type=float,
        help="The percentage of the labeled data to use for training.  The remainder is used for testing/validation.",
        default=0.8,
        required=False,
    )

    parser.add_argument(
        "--val_split",
        type=float,
        help="The percentage of the testing/validation data to allocate for validation.",
        default=0.5,
        required=False,
    )

    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Whether to skip downloading",
        required=False,
    )

    return parser.parse_args()


def main(args):
    import logging

    logging.basicConfig(
        level=logging.INFO,
    )
    logger = logging.getLogger("bench_cugraph_training")
    logger.setLevel(logging.INFO)

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    init_pytorch_worker(
        local_rank, use_rmm_torch_allocator=(args.framework == "cuGraph")
    )
    enable_spilling()
    print(f"worker initialized")
    dist.barrier()

    world_size = int(os.environ["SLURM_JOB_NUM_NODES"]) * args.gpus_per_node

    if args.use_wholegraph:
        # TODO support DGL too
        # TODO support WG without cuGraph
        if args.framework not in ["cuGraphPyG"]:
            raise ValueError("WG feature store only supported with cuGraph backends")
        from pylibwholegraph.torch.initialize import (
            get_global_communicator,
            get_local_node_communicator,
        )

        logger.info("initializing WG comms...")
        wm_comm = get_global_communicator()
        get_local_node_communicator()

        wm_comm = wm_comm.wmb_comm
        logger.info(f"rank {global_rank} successfully initialized WG comms")
        wm_comm.barrier()

    dataset = OGBNPapers100MDataset(
        replication_factor=args.replication_factor,
        dataset_dir=args.dataset_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        load_edge_index=(args.framework == "PyG"),
        backend="wholegraph" if args.use_wholegraph else "torch",
    )

    # Note: this does not generate WG files
    if global_rank == 0 and not args.skip_download:
        dataset.download()

    dist.barrier()

    fanout = [int(f) for f in args.fanout.split("_")]

    if args.framework == "PyG":
        from trainers.pyg import PyGNativeTrainer

        trainer = PyGNativeTrainer(
            model=args.model,
            dataset=dataset,
            device=local_rank,
            rank=global_rank,
            world_size=world_size,
            num_epochs=args.num_epochs,
            shuffle=True,
            replace=False,
            num_neighbors=fanout,
            batch_size=args.batch_size,
        )
    elif args.framework == "cuGraphPyG":
        sample_dir = os.path.join(
            args.sample_dir,
            f"ogbn_papers100M[{args.replication_factor}]_b{args.batch_size}_f{fanout}",
        )
        from trainers.pyg import PyGCuGraphTrainer

        trainer = PyGCuGraphTrainer(
            model=args.model,
            dataset=dataset,
            sample_dir=sample_dir,
            device=local_rank,
            rank=global_rank,
            world_size=world_size,
            num_epochs=args.num_epochs,
            shuffle=True,
            replace=False,
            num_neighbors=fanout,
            batch_size=args.batch_size,
            backend="wholegraph" if args.use_wholegraph else "torch",
        )
    elif args.framework == "cuGraphDGL":
        sample_dir = os.path.join(
            args.sample_dir,
            f"ogbn_papers100M[{args.replication_factor}]_b{args.batch_size}_f{fanout}",
        )
        from trainers.dgl import DGLCuGraphTrainer

        trainer = DGLCuGraphTrainer(
            model=args.model,
            dataset=dataset,
            sample_dir=sample_dir,
            device=local_rank,
            rank=global_rank,
            world_size=world_size,
            num_epochs=args.num_epochs,
            shuffle=True,
            replace=False,
            num_neighbors=[int(f) for f in args.fanout.split("_")],
            batch_size=args.batch_size,
        )
    else:
        raise ValueError("unsupported framework")

    logger.info(f"Trainer ready on rank {global_rank}")
    stats = trainer.train()
    logger.info(stats)

    with open(f"{args.output_file}[{global_rank}]", "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
