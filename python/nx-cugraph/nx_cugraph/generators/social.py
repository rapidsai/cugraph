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
import cupy as cp
import numpy as np

import nx_cugraph as nxcg

from ..utils import index_dtype, networkx_algorithm

__all__ = ["karate_club_graph"]


@networkx_algorithm
def karate_club_graph():
    # fmt: off
    src_indices = cp.array(
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5,
            6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10, 11, 12, 12, 13,
            13, 13, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 19, 20,
            20, 21, 21, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26,
            27, 27, 27, 27, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31,
            31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33,
            33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
        ],
        index_dtype,
    )
    dst_indices = cp.array(
        [
            1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13,
            17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0,
            6, 10, 0, 6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33,
            0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0,
            1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 23, 24, 31,
            29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24,
            25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13,
            14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32,
        ],
        index_dtype,
    )
    weights = cp.array(
        [
            4, 5, 3, 3, 3, 3, 2, 2, 2, 3, 1, 3, 2, 2, 2, 2, 4, 6, 3, 4, 5, 1, 2, 2,
            2, 5, 6, 3, 4, 5, 1, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 5, 3, 3,
            3, 2, 5, 3, 2, 4, 4, 3, 2, 5, 3, 3, 4, 1, 2, 2, 3, 3, 3, 1, 3, 3, 5, 3,
            3, 3, 3, 2, 3, 4, 3, 3, 2, 1, 1, 2, 2, 2, 1, 3, 1, 2, 2, 2, 3, 5, 4, 3,
            5, 4, 2, 3, 2, 5, 2, 7, 4, 2, 2, 4, 3, 4, 2, 2, 2, 3, 4, 4, 2, 2, 3, 3,
            3, 2, 2, 7, 2, 4, 4, 2, 3, 3, 3, 1, 3, 2, 5, 4, 3, 4, 5, 4, 2, 3, 2, 4,
            2, 1, 1, 3, 4, 2, 4, 2, 2, 3, 4, 5,
        ],
        np.int8,
    )
    # For now, cupy doesn't handle str dtypes and we only handle cupy arrays.
    # This means we are definitely cheating by using a numpy array here! FIXME
    clubs = np.array([
        "Mr. Hi", "Mr. Hi", "Mr. Hi", "Mr. Hi", "Mr. Hi", "Mr. Hi", "Mr. Hi",
        "Mr. Hi", "Mr. Hi", "Officer", "Mr. Hi", "Mr. Hi", "Mr. Hi", "Mr. Hi",
        "Officer", "Officer", "Mr. Hi", "Mr. Hi", "Officer", "Mr. Hi", "Officer",
        "Mr. Hi", "Officer", "Officer", "Officer", "Officer", "Officer", "Officer",
        "Officer", "Officer", "Officer", "Officer", "Officer", "Officer",
    ])
    # fmt: on
    return nxcg.Graph.from_coo(
        34,
        src_indices,
        dst_indices,
        edge_values={"weight": weights},
        node_values={"club": clubs},
        name="Zachary's Karate Club",
    )
