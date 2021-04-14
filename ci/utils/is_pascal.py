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

# FIXME: consolidate this code with ci/gpu/notebook_list.py

#
# Not strictly true... however what we mean is
# Pascal or earlier
#
pascal = False

device = cuda.get_current_device()
# check for the attribute using both pre and post numba 0.53 names
cc = getattr(device, 'COMPUTE_CAPABILITY', None) or \
     getattr(device, 'compute_capability')
if (cc[0] < 7):
    pascal = True

# Return zero (success) if pascal is True
if pascal:
    sys.exit(0)
else:
    sys.exit(1)
