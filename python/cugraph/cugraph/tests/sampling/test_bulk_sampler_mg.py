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
import dask_cudf
import cupy
import cugraph
from cugraph.datasets import karate
from cugraph.experimental import BulkSampler

import tempfile


@pytest.mark.mg
def test_bulk_sampler_simple(dask_client):
    el = karate.get_edgelist().reset_index().rename(columns={"index": "eid"})
    el["eid"] = el["eid"].astype("int32")
    el["etp"] = cupy.int32(0)

    G = cugraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(
        dask_cudf.from_cudf(el, npartitions=2),
        source="src",
        destination="dst",
        edge_attr=["wgt", "eid", "etp"],
    )

    tempdir_object = tempfile.TemporaryDirectory()
    bs = BulkSampler(
        batch_size=2,
        output_path=tempdir_object.name,
        graph=G,
        fanout_vals=[2, 2],
        with_replacement=False,
    )

    batches = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "start": cudf.Series([0, 5, 10, 15], dtype="int32"),
                "batch": cudf.Series([0, 0, 1, 1], dtype="int32"),
            }
        ),
        npartitions=2,
    )

    bs.add_batches(batches, start_col_name="start", batch_col_name="batch")
    bs.flush()

    recovered_samples = cudf.read_parquet(tempdir_object.name)

    for b in batches["batch"].unique().compute().values_host.tolist():
        assert b in recovered_samples["batch_id"].values_host.tolist()


@pytest.mark.mg
def test_bulk_sampler_mg_graph_sg_input(dask_client):
    el = karate.get_edgelist().reset_index().rename(columns={"index": "eid"})
    el["eid"] = el["eid"].astype("int32")
    el["etp"] = cupy.int32(0)

    G = cugraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(
        dask_cudf.from_cudf(el, npartitions=2),
        source="src",
        destination="dst",
        edge_attr=["wgt", "eid", "etp"],
    )

    tempdir_object = tempfile.TemporaryDirectory()
    bs = BulkSampler(
        batch_size=2,
        output_path=tempdir_object.name,
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

    recovered_samples = cudf.read_parquet(tempdir_object.name)

    for b in batches["batch"].unique().values_host.tolist():
        assert b in recovered_samples["batch_id"].values_host.tolist()
