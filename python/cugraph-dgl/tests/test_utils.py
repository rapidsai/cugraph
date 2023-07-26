# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import cudf
import cupy as cp
import numpy as np
import torch

from cugraph_dgl.dataloading.utils.sampling_helpers import (
    cast_to_tensor,
    _get_renumber_map,
    _split_tensor,
)


def test_casting_empty_array():
    ar = cp.zeros(shape=0, dtype=cp.int32)
    ser = cudf.Series(ar)
    output_tensor = cast_to_tensor(ser)
    assert output_tensor.dtype == torch.int32


def get_dummy_sampled_df():
    df = cudf.DataFrame()
    df["sources"] = [0, 0, 0, 0, 0, 0] + [np.nan] * 7
    df["destinations"] = [1, 2, 1, 2, 1, 2] + [np.nan] * 7
    df["batch_id"] = [0, 0, 1, 1, 2, 2] + [np.nan] * 7
    df["hop_id"] = [0, 1, 0, 1, 0, 1] + [np.nan] * 7
    print(len(df))
    df["map"] = [4, 7, 10, 13, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    df = df.astype("int32")
    df["hop_id"] = df["hop_id"].astype("uint8")
    df["map"] = df["map"].astype("int64")
    return df


def test_get_renumber_map():
    sampled_df = get_dummy_sampled_df()

    df, renumber_map, renumber_map_batch_indices = _get_renumber_map(sampled_df)

    # Ensure that map was dropped
    assert "map" not in df.columns

    expected_map = torch.as_tensor(
        [10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=torch.int32, device="cuda"
    )
    assert torch.equal(renumber_map, expected_map)

    expected_batch_indices = torch.as_tensor([3, 6], dtype=torch.int32, device="cuda")
    assert torch.equal(renumber_map_batch_indices, expected_batch_indices)

    # Ensure we dropped the Nans for rows  corresponding to the renumber_map
    assert len(df) == 6

    t_ls = _split_tensor(renumber_map, renumber_map_batch_indices)
    assert torch.equal(
        t_ls[0], torch.as_tensor([10, 11, 12], dtype=torch.int32, device="cuda")
    )
    assert torch.equal(
        t_ls[1], torch.as_tensor([13, 14, 15], dtype=torch.int32, device="cuda")
    )
    assert torch.equal(
        t_ls[2], torch.as_tensor([16, 17, 18], dtype=torch.int32, device="cuda")
    )
