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

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
os.environ["LIBCUDF_CUFILE_POLICY"] = "KVIKIO"
os.environ["KVIKIO_NTHREADS"] = "6"
import torch
from cugraph_dgl.dataloading.dataset import HomogenousBulkSamplerDataset
import time
import argparse
import json
import os
import cupy as cp
import numpy as np
import rmm
from multiprocessing import Manager


def setup_pool(gpu):
    # from rmm.allocators.torch import rmm_torch_allocator
    from rmm.allocators.cupy import rmm_cupy_allocator

    dev = cp.cuda.Device(gpu)
    dev.use()
    if gpu == 0:
        # torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
        pass
    rmm.reinitialize(pool_allocator=True, initial_pool_size=5e9, devices=[gpu])
    torch.cuda.set_device(gpu)
    cp.cuda.set_allocator(rmm_cupy_allocator)


def sampling_workflow(
    proc_id, devices, sampled_files, total_num_of_nodes, edge_dir, timing_dict
):
    dev_id = devices[proc_id]
    print(f"Setting up device = {dev_id}", flush=True)
    setup_pool(dev_id)
    current_files_to_process = sampled_files[proc_id]
    dist_init_method = "tcp://{master_ip}:{master_port}".format(
        master_ip="127.0.0.1", master_port="12345"
    )
    device = torch.device("cuda:" + str(dev_id))
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=dist_init_method,
        world_size=len(devices),
        rank=proc_id,
    )
    torch.distributed.barrier()
    # Warmup RUN
    run_dataloader(total_num_of_nodes, edge_dir, current_files_to_process)
    st = time.time()
    run_dataloader(total_num_of_nodes, edge_dir, current_files_to_process)
    et = time.time()
    time_taken = et - st
    timing_dict[proc_id] = time_taken
    torch.distributed.barrier()

    return


def run_dataloader(total_num_of_nodes, edge_dir, sampled_files):
    dataset = HomogenousBulkSamplerDataset(
        total_num_of_nodes, edge_dir=edge_dir, return_type="cugraph_dgl.nn.SparseGraph"
    )
    dataset.set_input_files(input_file_paths=sampled_files)
    dataloader = torch.utils.data.DataLoader(
        dataset, collate_fn=lambda x: x, shuffle=False, num_workers=0, batch_size=None
    )
    for input_nodes, output_nodes, blocks in dataloader:
        pass


# Function to Parse arguments
# sampling directory
# as a string
def parse_args():
    parser = argparse.ArgumentParser(prog="Run MFG Sampling Benchmark")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--num_training_gpus", type=int, required=False)

    args = parser.parse_args()
    args_d = {}
    sampling_dir = os.path.join(args.input_dir, "samples")
    meta_json = os.path.join(args.input_dir, "output_meta.json")
    with open(meta_json) as f:
        meta = json.load(f)
    args_d["total_num_nodes"] = meta["total_num_nodes"]
    args_d["total_num_edges"] = meta["total_num_edges"]
    args_d["batch_size"] = meta["batch_size"]
    args_d["cugraph_sampling_time"] = meta["execution_time"]
    args_d["sampling_dir"] = sampling_dir
    args_d["meta_dir"] = args.input_dir
    args_d["num_sampling_gpus"] = meta["num_sampling_gpus"]
    if args.num_training_gpus is None:
        num_training_gpus = torch.cuda.device_count()
    else:
        num_training_gpus = args.num_training_gpus
    args_d["num_training_gpus"] = num_training_gpus
    return args_d


def get_sampled_files(sampling_dir, num_sampling_gpus):
    input_file_paths = [os.path.join(sampling_dir, f) for f in os.listdir(sampling_dir)]
    return [input_file_paths[i::num_sampling_gpus] for i in range(num_sampling_gpus)]


def run_sampling_worflow(sampled_files, args_d):
    manager = Manager()
    print("Args dictionary is")
    print(args_d)
    print("--" * 30)
    total_num_nodes = args_d["total_num_nodes"]
    num_gpus = args_d["num_training_gpus"]

    import torch.multiprocessing as mp

    timing_dict = manager.dict()
    mp.spawn(
        sampling_workflow,
        args=(
            list(range(num_gpus)),
            sampled_files,
            total_num_nodes,
            "in",
            timing_dict,
        ),
        nprocs=num_gpus,
    )
    times = np.asarray([v for v in timing_dict.values()])
    print("Total time taken for sampling is {}".format(times.mean()))

    return times.mean()


if __name__ == "__main__":
    args_d = parse_args()
    num_training_gpus = args_d["num_training_gpus"]
    sampled_files = get_sampled_files(args_d["sampling_dir"], num_training_gpus)

    # First run is warmup
    run_sampling_worflow(sampled_files, args_d)
    # Actual timed run
    avg_mfg_time = run_sampling_worflow(sampled_files, args_d)
    output_meta = {}
    output_meta["total_num_nodes"] = args_d["total_num_nodes"]
    output_meta["total_num_edges"] = args_d["total_num_edges"]
    output_meta["sampling_batch_size"] = args_d["batch_size"]
    output_meta["cumulative_training_batch_size"] = (
        int(args_d["batch_size"]) * num_training_gpus
    )
    output_meta["num_sampling_gpus"] = args_d["num_sampling_gpus"]
    output_meta["num_training_gpus"] = num_training_gpus
    output_meta["cugraph_sampling_time"] = args_d["cugraph_sampling_time"]
    output_meta["cugraph_dgl_mfg_time"] = avg_mfg_time
    print(output_meta)

    with open(
        os.path.join(
            args_d["meta_dir"], f"{num_training_gpus}_training_gpus_output_meta.json"
        ),
        "w",
    ) as f:
        json.dump(output_meta, f)
