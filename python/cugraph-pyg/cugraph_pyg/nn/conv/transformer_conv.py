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

from .base import BaseConv


class TransformerConv(BaseConv):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        root_weight: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.root_weight = root_weight
        self.concat = concat
