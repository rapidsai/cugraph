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


@pytest.fixture
def example_scatter_data():
    src_feat = torch.Tensor([3, 1, 0, 1, 1, 2])
    dst_indices = torch.Tensor([0, 1, 2, 2, 3, 1])

    results = {
        "sum": torch.Tensor([3.0, 3.0, 1.0, 1.0]),
        "mean": torch.Tensor([3.0, 1.5, 0.5, 1.0]),
        "prod": torch.Tensor([3.0, 2.0, 0.0, 1.0]),
        "amax": torch.Tensor([3.0, 2.0, 1.0, 1.0]),
        "amin": torch.Tensor([3.0, 1.0, 0.0, 1.0]),
    }

    return src_feat, dst_indices, results
