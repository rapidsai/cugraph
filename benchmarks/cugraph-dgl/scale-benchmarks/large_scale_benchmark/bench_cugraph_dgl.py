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
import os

os.environ["LIBCUDF_CUFILE_POLICY"] = "KVIKIO"
os.environ["KVIKIO_NTHREADS"] = "16"
os.environ["RAPIDS_NO_INITIALIZE"] = "1"

import json
import argparse
import os
import json
import warnings

import torch
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
        help="Batch size (required for Native run)",
        required=False,
    )

    parser.add_argument(
        "--fanout",
        type=str,
        default="10_10_10",
        help="Fanout (required for Native run)",
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
        help="The framework to test (cuGraphDGL)",
        required=True,
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

    return parser.parse_args()


def main(args):
    import logging

    logging.basicConfig(
        level=logging.INFO,
    )
    logger = logging.getLogger("bench_cugraph_dgl")
    logger.setLevel(logging.INFO)

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    global_rank = int(os.getenv("RANK", 0))

    init_pytorch_worker(
        local_rank, use_rmm_torch_allocator=(args.framework == "cuGraphDGL")
    )
    enable_spilling()
    print(f"worker initialized")
    dist.barrier()

    # Have to import here to avoid creating CUDA context
    from trainers_cugraph import DGLCuGraphTrainer

    if os.getenv("SLURM_GPUS_PER_NODE", None) is None:
        world_size = int(os.getenv("LOCAL_WORLD_SIZE"))
    else:
        world_size = int(os.getenv("SLURM_JOB_NUM_NODES")) * int(
            os.getenv("SLURM_GPUS_PER_NODE")
        )

    print("world_size", world_size, flush=True)

    dataset = OGBNPapers100MDataset(
        replication_factor=args.replication_factor,
        dataset_dir=args.dataset_dir,
        train_split=args.train_split,
        val_split=args.val_split,
    )

    trainer = DGLCuGraphTrainer(
        model=args.model,
        dataset=dataset,
        sample_dir=args.sample_dir,
        device=local_rank,
        rank=global_rank,
        world_size=world_size,
        num_epochs=args.num_epochs,
        shuffle=True,
        replace=False,
        num_neighbors=[int(f) for f in args.fanout.split("_")],
        batch_size=args.batch_size,
    )
    stats = trainer.train()
    logger.info(stats)

    with open(f"{args.output_file}[{global_rank}].json", "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
