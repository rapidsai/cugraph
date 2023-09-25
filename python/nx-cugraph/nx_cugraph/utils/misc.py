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

import operator as op
import sys
from random import Random

import cupy as cp

__all__ = ["_groupby", "_handle_seed"]


def _groupby(groups: cp.ndarray, values: cp.ndarray) -> dict[int, cp.ndarray]:
    """Perform a groupby operation given an array of group IDs and array of values.

    Parameters
    ----------
    groups : cp.ndarray
        Array that holds the group IDs.
        Group IDs are assumed to be consecutive integers from 0.
    values : cp.ndarray
        Array of values to be grouped according to groups.
        Must be the same size as groups array.

    Returns
    -------
    dict with group IDs as keys and cp.ndarray as values.
    """
    # It would actually be easy to support groups that aren't consecutive integers,
    # but let's wait until we need it to implement it.
    sorted_groups = cp.argsort(groups)
    sorted_values = values[sorted_groups]
    rv = {}
    start = 0
    for i, end in enumerate(
        [*(cp.nonzero(cp.diff(groups[sorted_groups]))[0] + 1).tolist(), groups.size]
    ):
        rv[i] = sorted_values[start:end]
        start = end
    return rv


def _handle_seed(seed: int | Random | None) -> int:
    """Handle seed argument and ensure it is what pylibcugraph needs: an int."""
    if seed is None:
        return
    if isinstance(seed, Random):
        return seed.randint(0, sys.maxsize)
    return op.index(seed)  # Ensure seed is integral
