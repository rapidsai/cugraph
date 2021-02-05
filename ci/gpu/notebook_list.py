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

import re
import sys
import glob

from numba import cuda

#
# Not strictly true... however what we mean is
# Pascal or earlier
#
pascal = False

device = cuda.get_current_device()
cc = getattr(device, 'COMPUTE_CAPABILITY')
if (cc[0] < 7):
    pascal = True

for filename in glob.iglob('**/*.ipynb', recursive=True):
    skip = False
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

    if not skip:
        print(filename)
