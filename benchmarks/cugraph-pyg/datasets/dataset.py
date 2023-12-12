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

import torch
from typing import Dict, Tuple


class Dataset:
    @property
    def edge_index_dict(self) -> Dict[Tuple[str, str, str], Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    @property
    def x_dict(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    @property
    def y_dict(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    @property
    def train_dict(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    @property
    def test_dict(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    @property
    def val_dict(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    @property
    def num_input_features(self) -> int:
        raise NotImplementedError()

    @property
    def num_labels(self) -> int:
        raise NotImplementedError()

    def num_nodes(self, node_type: str) -> int:
        raise NotImplementedError()

    def num_edges(self, edge_type: Tuple[str, str, str]) -> int:
        raise NotImplementedError()
