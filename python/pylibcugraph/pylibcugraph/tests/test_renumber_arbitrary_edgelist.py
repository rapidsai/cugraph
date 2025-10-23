# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
