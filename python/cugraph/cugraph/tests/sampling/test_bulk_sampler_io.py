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
import tempfile
import os

from cugraph.gnn.data_loading.bulk_sampler_io import write_samples


@pytest.mark.sg
def test_bulk_sampler_io():
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

    offsets = cudf.DataFrame({"offsets": [0, 8], "batch_id": [0, 1]})

    tempdir_object = tempfile.TemporaryDirectory()
    write_samples(results, offsets, 1, tempdir_object.name)

    assert len(os.listdir(tempdir_object.name)) == 2

    df = cudf.read_parquet(os.path.join(tempdir_object.name, "batch=0-0.parquet"))
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

    df = cudf.read_parquet(os.path.join(tempdir_object.name, "batch=1-1.parquet"))
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
