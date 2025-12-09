# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import re
import argparse
import sys
import glob
from pathlib import Path

from cuda.bindings import runtime

# for adding another run type and skip file name add to this dictionary
runtype_dict = {
    "all": "",
    "ci": "SKIP_CI_TESTING",
    "nightly": "SKIP_NIGHTLY",
    "weekly": "SKIP_WEEKLY",
}


def skip_book_dir(runtype):
    # Add all run types here, currently only CI supported
    return runtype in runtype_dict and Path(runtype_dict.get(runtype)).is_file()


def _get_cuda_version_string():
    status, version = runtime.getLocalRuntimeVersion()
    if status != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError("Could not get CUDA runtime version.")
    major, minor = divmod(version, 1000)
    minor //= 10
    return f"{major}.{minor}"


cuda_version_string = _get_cuda_version_string()

parser = argparse.ArgumentParser(description="Condition for running the notebook tests")
parser.add_argument("runtype", type=str)

args = parser.parse_args()

runtype = args.runtype

if runtype not in runtype_dict:
    print(f"Unknown Run Type  = {runtype}", file=sys.stderr)
    exit()

for filename in glob.iglob("**/*.ipynb", recursive=True):
    skip = False
    if skip_book_dir(runtype):
        print(
            f"SKIPPING {filename} (Notebook skipped for run type {runtype}) "
            "due to skip file in folder.",
            file=sys.stderr,
        )
        skip = True
    else:
        for line in open(filename, "r"):
            if re.search("# Skip notebook test", line):
                skip = True
                print(f"SKIPPING {filename} (marked as skip)", file=sys.stderr)
                break
            elif re.search("dask", line):
                print(
                    f"SKIPPING {filename} (suspected Dask usage, not "
                    "currently automatable)",
                    file=sys.stderr,
                )
                skip = True
                break
            elif re.search("# Does not run on CUDA ", line) and (
                cuda_version_string in line
            ):
                print(
                    f"SKIPPING {filename} (does not run on CUDA {cuda_version_string})",
                    file=sys.stderr,
                )
                skip = True
                break
    if not skip:
        print(filename)
