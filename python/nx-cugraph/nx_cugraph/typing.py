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
from __future__ import annotations

from collections.abc import Hashable
from typing import TypeVar

import cupy as cp
import numpy as np

AttrKey = TypeVar("AttrKey", bound=Hashable)
EdgeKey = TypeVar("EdgeKey", bound=Hashable)
NodeKey = TypeVar("NodeKey", bound=Hashable)
EdgeTuple = tuple[NodeKey, NodeKey]
EdgeValue = TypeVar("EdgeValue")
NodeValue = TypeVar("NodeValue")
IndexValue = TypeVar("IndexValue")
Dtype = TypeVar("Dtype")


class any_ndarray:
    def __class_getitem__(cls, item):
        return cp.ndarray[item] | np.ndarray[item]
