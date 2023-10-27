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
from typing import SupportsIndex

import cupy as cp
import numpy as np

__all__ = ["index_dtype", "_groupby", "_seed_to_int", "_get_int_dtype"]

# This may switch to np.uint32 at some point
index_dtype = np.int32


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


def _seed_to_int(seed: int | Random | None) -> int:
    """Handle any valid seed argument and convert it to an int if necessary."""
    if seed is None:
        return
    if isinstance(seed, Random):
        return seed.randint(0, sys.maxsize)
    return op.index(seed)  # Ensure seed is integral


def _get_int_dtype(
    val: SupportsIndex, *, signed: bool | None = None, unsigned: bool | None = None
):
    """Determine the smallest integer dtype that can store the integer ``val``.

    If signed or unsigned are unspecified, then signed integers are preferred
    unless the value can be represented by a smaller unsigned integer.

    Raises
    ------
    ValueError : If the value cannot be represented with an int dtype.
    """
    # This is similar in spirit to `np.min_scalar_type`
    if signed is not None:
        if unsigned is not None and (not signed) is (not unsigned):
            raise TypeError(
                f"signed (={signed}) and unsigned (={unsigned}) keyword arguments "
                "are incompatible."
            )
        signed = bool(signed)
        unsigned = not signed
    elif unsigned is not None:
        unsigned = bool(unsigned)
        signed = not unsigned

    val = op.index(val)  # Ensure val is integral
    if val < 0:
        if unsigned:
            raise ValueError(f"Value is incompatible with unsigned int: {val}.")
        signed = True
        unsigned = False

    if signed is not False:
        # Number of bytes (and a power of two)
        signed_nbytes = (val + (val < 0)).bit_length() // 8 + 1
        signed_nbytes = next(
            filter(
                signed_nbytes.__le__,
                itertools.accumulate(itertools.repeat(2), op.mul, initial=1),
            )
        )
    if unsigned is not False:
        # Number of bytes (and a power of two)
        unsigned_nbytes = (val.bit_length() + 7) // 8
        unsigned_nbytes = next(
            filter(
                unsigned_nbytes.__le__,
                itertools.accumulate(itertools.repeat(2), op.mul, initial=1),
            )
        )
        if signed is None and unsigned is None:
            # Prefer signed int if same size
            signed = signed_nbytes <= unsigned_nbytes

    if signed:
        dtype_string = f"i{signed_nbytes}"
    else:
        dtype_string = f"u{unsigned_nbytes}"
    try:
        return np.dtype(dtype_string)
    except TypeError as exc:
        raise ValueError("Value is too large to store as integer: {val}") from exc
