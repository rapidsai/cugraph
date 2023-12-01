# Copyright (c) 2021, NVIDIA CORPORATION.
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
import sys
import time
import yaml

from dask.distributed import Client


def initialize_dask_cuda(communication_type):
    communication_type = communication_type.lower()
    if "ucx" in communication_type:
        os.environ["UCX_MAX_RNDV_RAILS"] = "1"

    if communication_type == "ucx-ib":
        os.environ["UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES"]="cuda"
        os.environ["DASK_RMM__POOL_SIZE"]="0.5GB"
        os.environ["DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT"]="True"


def wait_for_workers(
    num_expected_workers, scheduler_file_path, communication_type, timeout_after=0
):
    """
    Waits until num_expected_workers workers are available based on
    the workers managed by scheduler_file_path, then returns 0. If
    timeout_after is specified, will return 1 if num_expected_workers
    workers are not available before the timeout.
    """
    # FIXME: use scheduler file path from global environment if none
    # supplied in configuration yaml

    print("wait_for_workers.py - initializing client...", end="")
    sys.stdout.flush()
    initialize_dask_cuda(communication_type)
    print("done.")
    sys.stdout.flush()

    ready = False
    start_time = time.time()
    while not ready:
        if timeout_after and ((time.time() - start_time) >= timeout_after):
            print(
                f"wait_for_workers.py timed out after {timeout_after} seconds before finding {num_expected_workers} workers."
            )
            sys.stdout.flush()
            break
        with Client(scheduler_file=scheduler_file_path) as client:
            num_workers = len(client.scheduler_info()["workers"])
            if num_workers < num_expected_workers:
                print(
                    f"wait_for_workers.py expected {num_expected_workers} but got {num_workers}, waiting..."
                )
                sys.stdout.flush()
                time.sleep(5)
            else:
                print(f"wait_for_workers.py got {num_workers} workers, done.")
                sys.stdout.flush()
                ready = True

    if ready is False:
        return 1
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--num-expected-workers",
        type=int,
        required=False,
        help="Number of workers to wait for. If not specified, "
        "uses the NUM_WORKERS env var if set, otherwise defaults "
        "to 16.",
    )
    ap.add_argument(
        "--scheduler-file-path",
        type=str,
        required=True,
        help="Path to shared scheduler file to read.",
    )
    ap.add_argument(
        "--communication-type",
        type=str,
        default="tcp",
        required=False,
        help="Initiliaze dask_cuda based on the cluster communication type."
        "Supported values are tcp(default), ucx, ucxib, ucx-ib.",
    )
    ap.add_argument(
        "--timeout-after",
        type=int,
        default=0,
        required=False,
        help="Number of seconds to wait for workers. "
        "Default is 0 which means wait forever.",
    )
    args = ap.parse_args()

    if args.num_expected_workers is None:
        args.num_expected_workers = os.environ.get("NUM_WORKERS", 16)

    exitcode = wait_for_workers(
        num_expected_workers=args.num_expected_workers,
        scheduler_file_path=args.scheduler_file_path,
        communication_type=args.communication_type,
        timeout_after=args.timeout_after,
    )

    sys.exit(exitcode)
