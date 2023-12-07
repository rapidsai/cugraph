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

import os
import shutil

import pytest

import cupy
import cudf
from cugraph.gnn.data_loading.bulk_sampler_io import write_samples
from cugraph.utilities.utils import create_directory_with_overwrite


@pytest.mark.sg
def test_bulk_sampler_io(scratch_dir):
    results = cudf.DataFrame(
        {
            "sources": [0, 0, 1, 2, 2, 2, 3, 4, 5, 5, 6, 7],
            "destinations": [1, 2, 3, 3, 3, 4, 1, 1, 6, 7, 2, 3],
            "edge_id": None,
            "edge_type": None,
            "weight": None,
            "hop_id": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        }
    )

    assert len(results) == 12

    offsets = cudf.DataFrame({"offsets": [0, 8, 12], "batch_id": [0, 1, None]})

    samples_path = os.path.join(scratch_dir, "test_bulk_sampler_io")
    create_directory_with_overwrite(samples_path)

    write_samples(results, offsets, None, 1, samples_path)

    assert len(os.listdir(samples_path)) == 2

    df = cudf.read_parquet(os.path.join(samples_path, "batch=0-0.parquet"))
    assert len(df) == 8

    assert (
        df.sources.values_host.tolist()
        == results.sources.iloc[0:8].values_host.tolist()
    )
    assert (
        df.destinations.values_host.tolist()
        == results.destinations.iloc[0:8].values_host.tolist()
    )
    assert (
        df.hop_id.values_host.tolist() == results.hop_id.iloc[0:8].values_host.tolist()
    )
    assert (df.batch_id == 0).all()

    df = cudf.read_parquet(os.path.join(samples_path, "batch=1-1.parquet"))
    assert len(df) == 4
    assert (
        df.sources.values_host.tolist()
        == results.sources.iloc[8:12].values_host.tolist()
    )
    assert (
        df.destinations.values_host.tolist()
        == results.destinations.iloc[8:12].values_host.tolist()
    )
    assert (
        df.hop_id.values_host.tolist() == results.hop_id.iloc[8:12].values_host.tolist()
    )
    assert (df.batch_id == 1).all()

    shutil.rmtree(samples_path)


@pytest.mark.sg
def test_bulk_sampler_io_empty_batch(scratch_dir):
    sources_array = [
        0,
        0,
        1,
        2,
        2,
        2,
        3,
        4,
        5,
        5,
        6,
        7,
        9,
        9,
        12,
        13,
        29,
        29,
        31,
        14,
    ]

    destinations_array = [
        1,
        2,
        3,
        3,
        3,
        4,
        1,
        1,
        6,
        7,
        2,
        3,
        12,
        13,
        18,
        19,
        31,
        14,
        15,
        16,
    ]

    hops_array = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]

    results = cudf.DataFrame(
        {
            "sources": sources_array,
            "destinations": destinations_array,
            "edge_id": None,
            "edge_type": None,
            "weight": None,
            "hop_id": hops_array,
        }
    )

    assert len(results) == 20

    # some batches are missing
    offsets = cudf.DataFrame(
        {"offsets": [0, 8, 12, 16, 20], "batch_id": [0, 3, 4, 10, None]}
    )

    samples_path = os.path.join(scratch_dir, "test_bulk_sampler_io_empty_batch")
    create_directory_with_overwrite(samples_path)

    write_samples(results, offsets, None, 2, samples_path)

    files = os.listdir(samples_path)
    assert len(files) == 2

    df0 = cudf.read_parquet(os.path.join(samples_path, "batch=0-1.parquet"))

    assert df0.batch_id.min() == 0
    assert df0.batch_id.max() == 1

    df1 = cudf.read_parquet(os.path.join(samples_path, "batch=4-5.parquet"))
    assert df1.batch_id.min() == 4
    assert df1.batch_id.max() == 5

    shutil.rmtree(samples_path)


@pytest.mark.sg
def test_bulk_sampler_io_mock_csr(scratch_dir):
    major_offsets_array = cudf.Series([0, 5, 10, 15])
    minors_array = cudf.Series([1, 2, 3, 4, 8, 9, 1, 3, 4, 5, 3, 0, 4, 9, 1])
    edge_ids = cudf.Series(cupy.arange(len(minors_array)))

    # 2 hops
    label_hop_offsets = cudf.Series([0, 1, 3])

    # map
    renumber_map = cudf.Series(cupy.arange(10))
    renumber_map_offsets = cudf.Series([0, 10])

    results_df = cudf.DataFrame()
    results_df["minors"] = minors_array
    results_df["major_offsets"] = major_offsets_array
    results_df["edge_id"] = edge_ids
    results_df["edge_type"] = None
    results_df["weight"] = None

    offsets_df = cudf.DataFrame()
    offsets_df["offsets"] = label_hop_offsets
    offsets_df["renumber_map_offsets"] = renumber_map_offsets
    offsets_df["batch_id"] = cudf.Series([0])

    renumber_df = cudf.DataFrame()
    renumber_df["map"] = renumber_map

    samples_path = os.path.join(scratch_dir, "test_bulk_sampler_io_mock_csr")
    create_directory_with_overwrite(samples_path)

    write_samples(results_df, offsets_df, renumber_df, 1, samples_path)

    result = cudf.read_parquet(os.path.join(samples_path, "batch=0-0.parquet"))

    assert (
        result.minors.dropna().values_host.tolist() == minors_array.values_host.tolist()
    )
    assert (
        result.major_offsets.dropna().values_host.tolist()
        == major_offsets_array.values_host.tolist()
    )
    assert result.edge_id.dropna().values_host.tolist() == edge_ids.values_host.tolist()
    assert (
        result.renumber_map_offsets.dropna().values_host.tolist()
        == renumber_map_offsets.values_host.tolist()
    )
    assert result.map.dropna().values_host.tolist() == renumber_map.values_host.tolist()
    assert (
        result.label_hop_offsets.dropna().values_host.tolist()
        == label_hop_offsets.values_host.tolist()
    )

    shutil.rmtree(samples_path)
