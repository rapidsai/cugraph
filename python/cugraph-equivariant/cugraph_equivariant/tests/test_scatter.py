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

import pytest
import torch
from cugraph_equivariant.utils import scatter_reduce

@pytest.mark.parametrize("reduce", ["sum", "mean", "prod", "amax", "amin"])
def test_scatter_reduce(example_scatter_data, reduce):
    device = torch.device("cuda:0")
    src, index, out_true = example_scatter_data
    src = src.to(device)
    index = index.to(device)

    out = scatter_reduce(src, index, dim=0, dim_size=None, reduce=reduce)

    assert torch.allclose(out.cpu(), out_true[reduce])
