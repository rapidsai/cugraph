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

from typing import Union, List


def extend_tensor(t: Union[List[int], torch.Tensor], l: int):
    t = torch.as_tensor(t)

    return torch.concat([t, torch.zeros(l - len(t), dtype=t.dtype, device=t.device)])


class Trainer:
    @property
    def rank(self):
        raise NotImplementedError()

    @property
    def model(self):
        raise NotImplementedError()

    @property
    def dataset(self):
        raise NotImplementedError()

    @property
    def data(self):
        raise NotImplementedError()

    @property
    def optimizer(self):
        raise NotImplementedError()

    @property
    def num_epochs(self) -> int:
        raise NotImplementedError()

    def get_loader(self, epoch: int = 0, stage="train"):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()
