# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import cudf
import cupy as cp
import torch

from cugraph_dgl.dataloading.utils.sampling_helpers import cast_to_tensor


def test_casting_empty_array():
    ar = cp.zeros(shape=0, dtype=cp.int32)
    ser = cudf.Series(ar)
    output_tensor = cast_to_tensor(ser)
    assert output_tensor.dtype == torch.int32
