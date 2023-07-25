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

import pytest
import cudf
import cupy
import cugraph
from cugraph.experimental.datasets import karate
from cugraph.experimental.gnn import BulkSampler
from cugraph.utilities.utils import create_directory_with_overwrite

import os
import shutil
import re


@pytest.mark.sg
def test_bulk_sampler_simple(scratch_dir):
    el = karate.get_edgelist().reset_index().rename(columns={"index": "eid"})
    el["eid"] = el["eid"].astype("int32")
    el["etp"] = cupy.int32(0)

    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(
        el,
        source="src",
        destination="dst",
        edge_attr=["wgt", "eid", "etp"],
    )

    samples_path = os.path.join(scratch_dir, "test_bulk_sampler_simple")
    create_directory_with_overwrite(samples_path)

    bs = BulkSampler(
        batch_size=2,
        output_path=samples_path,
        graph=G,
        fanout_vals=[2, 2],
        with_replacement=False,
    )

    batches = cudf.DataFrame(
        {
            "start": cudf.Series([0, 5, 10, 15], dtype="int32"),
            "batch": cudf.Series([0, 0, 1, 1], dtype="int32"),
        }
    )

    bs.add_batches(batches, start_col_name="start", batch_col_name="batch")
    bs.flush()

    recovered_samples = cudf.read_parquet(samples_path)
    assert "map" not in recovered_samples.columns

    for b in batches["batch"].unique().values_host.tolist():
        assert b in recovered_samples["batch_id"].values_host.tolist()

    shutil.rmtree(samples_path)


@pytest.mark.sg
def test_bulk_sampler_remainder(scratch_dir):
    el = karate.get_edgelist().reset_index().rename(columns={"index": "eid"})
    el["eid"] = el["eid"].astype("int32")
    el["etp"] = cupy.int32(0)

    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(
        el,
        source="src",
        destination="dst",
        edge_attr=["wgt", "eid", "etp"],
    )

    samples_path = os.path.join(scratch_dir, "test_bulk_sampler_remainder")
    create_directory_with_overwrite(samples_path)

    bs = BulkSampler(
        batch_size=2,
        output_path=samples_path,
        graph=G,
        seeds_per_call=7,
        batches_per_partition=2,
        fanout_vals=[2, 2],
        with_replacement=False,
    )

    # Should process batch (0, 1, 2) then (3, 4, 5) then 6

    batches = cudf.DataFrame(
        {
            "start": cudf.Series(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype="int32"
            ),
            "batch": cudf.Series(
                [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype="int32"
            ),
        }
    )

    bs.add_batches(batches, start_col_name="start", batch_col_name="batch")
    bs.flush()

    recovered_samples = cudf.read_parquet(samples_path)
    assert "map" not in recovered_samples.columns

    for b in batches["batch"].unique().values_host.tolist():
        assert b in recovered_samples["batch_id"].values_host.tolist()

    for x in range(0, 6, 2):
        subdir = f"{x}-{x+1}"
        df = cudf.read_parquet(os.path.join(samples_path, f"batch={subdir}.parquet"))

        assert ((df.batch_id == x) | (df.batch_id == (x + 1))).all()
        assert ((df.hop_id == 0) | (df.hop_id == 1)).all()

    assert (
        cudf.read_parquet(os.path.join(samples_path, "batch=6-6.parquet")).batch_id == 6
    ).all()

    shutil.rmtree(samples_path)


@pytest.mark.sg
def test_bulk_sampler_large_batch_size(scratch_dir):
    el = karate.get_edgelist().reset_index().rename(columns={"index": "eid"})
    el["eid"] = el["eid"].astype("int32")
    el["etp"] = cupy.int32(0)

    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(
        el,
        source="src",
        destination="dst",
        edge_attr=["wgt", "eid", "etp"],
    )

    samples_path = os.path.join(scratch_dir, "test_bulk_sampler_large_batch_size")
    if os.path.exists(samples_path):
        shutil.rmtree(samples_path)
    os.makedirs(samples_path)
    bs = BulkSampler(
        batch_size=5120,
        output_path=samples_path,
        graph=G,
        fanout_vals=[2, 2],
        with_replacement=False,
    )

    batches = cudf.DataFrame(
        {
            "start": cudf.Series([0, 5, 10, 15], dtype="int32"),
            "batch": cudf.Series([0, 0, 1, 1], dtype="int32"),
        }
    )

    bs.add_batches(batches, start_col_name="start", batch_col_name="batch")
    bs.flush()

    recovered_samples = cudf.read_parquet(samples_path)
    assert "map" not in recovered_samples.columns

    for b in batches["batch"].unique().values_host.tolist():
        assert b in recovered_samples["batch_id"].values_host.tolist()

    shutil.rmtree(samples_path)


@pytest.mark.sg
def test_bulk_sampler_partitions(scratch_dir):
    el = karate.get_edgelist().reset_index().rename(columns={"index": "eid"})
    el["eid"] = el["eid"].astype("int32")
    el["etp"] = cupy.int32(0)

    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(
        el,
        source="src",
        destination="dst",
        edge_attr=["wgt", "eid", "etp"],
    )

    samples_path = os.path.join(scratch_dir, "test_bulk_sampler_partitions")
    if os.path.exists(samples_path):
        shutil.rmtree(samples_path)
    os.makedirs(samples_path)

    bs = BulkSampler(
        batch_size=3,
        output_path=samples_path,
        graph=G,
        fanout_vals=[2, 2],
        with_replacement=False,
        batches_per_partition=2,
        renumber=True,
    )

    batches = cudf.DataFrame(
        {
            "start": cudf.Series([0, 5, 6, 10, 15, 17, 18, 9, 23], dtype="int32"),
            "batch": cudf.Series([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype="int32"),
        }
    )

    bs.add_batches(batches, start_col_name="start", batch_col_name="batch")
    bs.flush()

    for file in os.listdir(samples_path):
        start_batch_id, end_batch_id = [
            int(x) for x in re.match(r"batch=([0-9]+)-([0-9]+).parquet", file).groups()
        ]

        recovered_samples = cudf.read_parquet(os.path.join(samples_path, file))
        recovered_map = recovered_samples.map
        recovered_samples = recovered_samples.drop("map", axis=1).dropna()

        for current_batch_id in range(start_batch_id, end_batch_id + 1):
            map_start_ix = recovered_map.iloc[current_batch_id - start_batch_id]
            map_end_ix = recovered_map.iloc[current_batch_id - start_batch_id + 1]
            map_current_batch = recovered_map.iloc[map_start_ix:map_end_ix]
            n_unique = cudf.concat(
                [
                    recovered_samples[
                        recovered_samples.batch_id == current_batch_id
                    ].sources,
                    recovered_samples[
                        recovered_samples.batch_id == current_batch_id
                    ].destinations,
                ]
            ).nunique()
            assert len(map_current_batch) == n_unique
