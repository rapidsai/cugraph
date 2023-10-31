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

import itertools
import operator as op
import sys
from random import Random

import cupy as cp

try:
    from itertools import pairwise  # Python >=3.10
except ImportError:

    def pairwise(it):
        it = iter(it)
        for prev in it:
            for cur in it:
                yield (prev, cur)
                prev = cur


__all__ = ["_groupby", "_seed_to_int"]


def _groupby(
    groups: cp.ndarray, values: cp.ndarray, groups_are_canonical: bool = False
) -> dict[int, cp.ndarray]:
    """Perform a groupby operation given an array of group IDs and array of values.

    Parameters
    ----------
    groups : cp.ndarray
        Array that holds the group IDs.
    values : cp.ndarray
        Array of values to be grouped according to groups.
        Must be the same size as groups array.
    groups_are_canonical : bool, default False
        Whether the group IDs are consecutive integers beginning with 0.

    Returns
    -------
    dict with group IDs as keys and cp.ndarray as values.
    """
    if groups.size == 0:
        return {}
    sort_indices = cp.argsort(groups)
    sorted_groups = groups[sort_indices]
    sorted_values = values[sort_indices]
    prepend = 1 if groups_are_canonical else sorted_groups[0] + 1
    left_bounds = cp.nonzero(cp.diff(sorted_groups, prepend=prepend))[0]
    boundaries = pairwise(itertools.chain(left_bounds.tolist(), [groups.size]))
    if groups_are_canonical:
        it = enumerate(boundaries)
    else:
        it = zip(sorted_groups[left_bounds].tolist(), boundaries)
    return {group: sorted_values[start:end] for group, (start, end) in it}


def _seed_to_int(seed: int | Random | None) -> int:
    """Handle any valid seed argument and convert it to an int if necessary."""
    if seed is None:
        return
    if isinstance(seed, Random):
        return seed.randint(0, sys.maxsize)
    return op.index(seed)  # Ensure seed is integral
