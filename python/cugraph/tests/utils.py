# Copyright (c) 2019, NVIDIA CORPORATION.
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

from scipy.io import mmread

import cudf


def read_mtx_file(mm_file):
    print('Reading ' + str(mm_file) + '...')
    return mmread(mm_file).asfptype()


def read_csv_file(mm_file, read_weights_in_sp=True):
    print('Reading ' + str(mm_file) + '...')
    if read_weights_in_sp is True:
        return cudf.read_csv(mm_file, delimiter=' ',
                             dtype=['int32', 'int32', 'float32'], header=None)
    else:
        return cudf.read_csv(mm_file, delimiter=' ',
                             dtype=['int32', 'int32', 'float64'], header=None)
