# Copyright (c) 2025, NVIDIA CORPORATION.
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

import numpy as np
import cupy

from pylibcugraph import renumber_arbitrary_edgelist
from pylibcugraph.resource_handle import ResourceHandle


def test_renumber_arbitrary_edgelist():
    renumber_map = np.array([5, 6, 1, 4, 0, 9])
    srcs = cupy.array([1, 1, 4, 4, 5, 5, 0, 9])
    dsts = cupy.array([6, 4, 5, 1, 4, 6, 1, 0])

    renumber_arbitrary_edgelist(
        ResourceHandle(),
        renumber_map,
        srcs,
        dsts,
    )

    assert srcs.tolist() == [2, 2, 3, 3, 0, 0, 4, 5]
    assert dsts.tolist() == [1, 3, 0, 2, 3, 1, 2, 4]
