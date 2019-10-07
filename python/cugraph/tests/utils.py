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

from scipy.sparse import coo_matrix

import cudf
import pandas as pd


def read_csv_for_nx(csv_file, read_weights_in_sp=True):
    print('Reading ' + str(csv_file) + '...')
    if read_weights_in_sp is True:
        df = pd.read_csv(csv_file, delimiter=' ', header=None,
                         names=['0', '1', '2'],
                         dtype={'0': 'int32', '1': 'int32', '2': 'float32'})
    else:
        df = pd.read_csv(csv_file, delimiter=' ', header=None,
                         names=['0', '1', '2'],
                         dtype={'0': 'int32', '1': 'int32', '2': 'float64'})

    nverts = 1 + max(df['0'].max(), df['1'].max())

    return coo_matrix((df['2'], (df['0'], df['1'])), shape=(nverts, nverts))


def read_csv_file(csv_file, read_weights_in_sp=True):
    print('Reading ' + str(csv_file) + '...')
    if read_weights_in_sp is True:
        return cudf.read_csv(csv_file, delimiter=' ',
                             dtype=['int32', 'int32', 'float32'], header=None)
    else:
        return cudf.read_csv(csv_file, delimiter=' ',
                             dtype=['int32', 'int32', 'float64'], header=None)
