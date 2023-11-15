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
import numpy as np
import pytest

from nx_cugraph.utils import _get_int_dtype


def test_get_int_dtype():
    uint8 = np.dtype(np.uint8)
    uint16 = np.dtype(np.uint16)
    uint32 = np.dtype(np.uint32)
    uint64 = np.dtype(np.uint64)
    # signed
    cur = np.iinfo(np.int8)
    for val in [cur.min, cur.min + 1, -1, 0, 1, cur.max - 1, cur.max]:
        assert _get_int_dtype(val) == np.int8
        assert _get_int_dtype(val, signed=True) == np.int8
        if val >= 0:
            assert _get_int_dtype(val, unsigned=True) == np.uint8
            assert _get_int_dtype(val + 1, unsigned=True) == np.uint8
    prev = cur
    cur = np.iinfo(np.int16)
    for val in [cur.min, cur.min + 1, prev.min - 1, prev.max + 1, cur.max - 1, cur.max]:
        assert _get_int_dtype(val) != prev.dtype
        assert _get_int_dtype(val, signed=True) == np.int16
        if val >= 0:
            assert _get_int_dtype(val, unsigned=True) in {uint8, uint16}
            assert _get_int_dtype(val + 1, unsigned=True) in {uint8, uint16}
    prev = cur
    cur = np.iinfo(np.int32)
    for val in [cur.min, cur.min + 1, prev.min - 1, prev.max + 1, cur.max - 1, cur.max]:
        assert _get_int_dtype(val) != prev.dtype
        assert _get_int_dtype(val, signed=True) == np.int32
        if val >= 0:
            assert _get_int_dtype(val, unsigned=True) in {uint16, uint32}
            assert _get_int_dtype(val + 1, unsigned=True) in {uint16, uint32}
    prev = cur
    cur = np.iinfo(np.int64)
    for val in [cur.min, cur.min + 1, prev.min - 1, prev.max + 1, cur.max - 1, cur.max]:
        assert _get_int_dtype(val) != prev.dtype
        assert _get_int_dtype(val, signed=True) == np.int64
        if val >= 0:
            assert _get_int_dtype(val, unsigned=True) in {uint32, uint64}
            assert _get_int_dtype(val + 1, unsigned=True) in {uint32, uint64}
    with pytest.raises(ValueError, match="Value is too"):
        _get_int_dtype(cur.min - 1, signed=True)
    with pytest.raises(ValueError, match="Value is too"):
        _get_int_dtype(cur.max + 1, signed=True)

    # unsigned
    cur = np.iinfo(np.uint8)
    for val in [0, 1, cur.max - 1, cur.max]:
        assert _get_int_dtype(val) == (np.uint8 if val > 1 else np.int8)
        assert _get_int_dtype(val, unsigned=True) == np.uint8
    assert _get_int_dtype(cur.max + 1) == np.int16
    cur = np.iinfo(np.uint16)
    for val in [cur.max - 1, cur.max]:
        assert _get_int_dtype(val, unsigned=True) == np.uint16
    assert _get_int_dtype(cur.max + 1) == np.int32
    cur = np.iinfo(np.uint32)
    for val in [cur.max - 1, cur.max]:
        assert _get_int_dtype(val, unsigned=True) == np.uint32
    assert _get_int_dtype(cur.max + 1) == np.int64
    cur = np.iinfo(np.uint64)
    for val in [cur.max - 1, cur.max]:
        assert _get_int_dtype(val, unsigned=True) == np.uint64
    with pytest.raises(ValueError, match="Value is incompatible"):
        _get_int_dtype(cur.min - 1, unsigned=True)
    with pytest.raises(ValueError, match="Value is too"):
        _get_int_dtype(cur.max + 1, unsigned=True)

    # API
    with pytest.raises(TypeError, match="incompatible"):
        _get_int_dtype(7, signed=True, unsigned=True)
    assert _get_int_dtype(7, signed=True, unsigned=False) == np.int8
    assert _get_int_dtype(7, signed=False, unsigned=True) == np.uint8
