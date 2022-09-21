# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import re
import argparse
import sys
import glob
import os
from pathlib import Path

from numba import cuda

# for adding another run type and skip file name add to this dictionary
runtype_dict = {"all" : "", "ci" : "SKIP_CI_TESTING", 
                "nightly" : "SKIP_NIGHTLY", "weekly" : "SKIP_WEEKLY"}


def skip_book_dir(runtype):
    # Add all run types here, currently only CI supported

    if runtype in runtype_dict.keys():
        if Path(runtype_dict.get(runtype)).is_file():
            return True
    return False


cuda_version_string = ".".join([str(n) for n in cuda.runtime.get_version()])
#
# Not strictly true... however what we mean is
# Pascal or earlier
#
pascal = False
ampere = False
device = cuda.get_current_device()

parser = argparse.ArgumentParser(description='Condition for running the notebook tests')
parser.add_argument('runtype', type=str)

args = parser.parse_args()

runtype = args.runtype

if runtype not in runtype_dict.keys():
    print(f'Unknown Run Type  = {runtype}', file=sys.stderr)
    exit()


# check for the attribute using both pre and post numba 0.53 names
cc = getattr(device, 'COMPUTE_CAPABILITY', None) or \
     getattr(device, 'compute_capability')
if (cc[0] < 7):
    pascal = True
if (cc[0] >= 8):
    ampere = True

skip = False
for filename in glob.iglob('**/*.ipynb', recursive=True):
    skip = False
    if (skip_book_dir(runtype) == True):
        print(f'SKIPPING {filename} (Notebook skipped for run type {runtype}) due to skip file in folder.', file=sys.stderr)
        skip = True
    else:
        for line in open(filename, 'r'):
            if re.search('# Skip notebook test', line):
                skip = True
                print(f'SKIPPING {filename} (marked as skip)', file=sys.stderr)
                break;
            elif re.search('dask', line):
                print(f'SKIPPING {filename} (suspected Dask usage, not currently automatable)', file=sys.stderr)
                skip = True
                break;
            elif pascal and re.search('# Does not run on Pascal', line):
                print(f'SKIPPING {filename} (does not run on Pascal)', file=sys.stderr)
                skip = True
                break;
            elif ampere and re.search('# Does not run on Ampere', line):
                print(f'SKIPPING {filename} (does not run on Ampere)', file=sys.stderr)
                skip = True
                break;
            elif re.search('# Does not run on CUDA ', line) and \
             (cuda_version_string in line):
                print(f'SKIPPING {filename} (does not run on CUDA {cuda_version_string})',
                  file=sys.stderr)
                skip = True
                break;
    if not skip:
        print(filename)
