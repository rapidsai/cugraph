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

import tempfile
import os

import cudf
import cupy

from cugraph_pyg.loader import CuGraphNeighborLoader
from cugraph_pyg.loader import BulkSampleLoader
from cugraph_pyg.data import CuGraphStore
from cugraph.gnn import FeatureStore
from cugraph.utilities.utils import import_optional, MissingModule

torch = import_optional("torch")


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_cugraph_loader_basic(karate_gnn):
    F, G, N = karate_gnn
    cugraph_store = CuGraphStore(F, G, N)
    loader = CuGraphNeighborLoader(
        (cugraph_store, cugraph_store),
        torch.arange(N["type0"] + N["type1"], dtype=torch.int64),
        10,
        num_neighbors=[4, 4],
        random_state=62,
        replace=False,
    )

    samples = [s for s in loader]

    assert len(samples) == 3
    for sample in samples:
        if "type0" in sample:
            for prop in sample["type0"]["prop0"].tolist():
                assert prop % 31 == 0

        if "type1" in sample:
            for prop in sample["type1"]["prop0"].tolist():
                assert prop % 41 == 0


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_cugraph_loader_hetero(karate_gnn):
    F, G, N = karate_gnn
    cugraph_store = CuGraphStore(F, G, N)
    loader = CuGraphNeighborLoader(
        (cugraph_store, cugraph_store),
        input_nodes=("type1", torch.tensor([0, 1, 2, 5], device="cuda")),
        batch_size=2,
        num_neighbors=[4, 4],
        random_state=62,
        replace=False,
    )

    samples = [s for s in loader]

    assert len(samples) == 2
    for sample in samples:
        if "type0" in sample:
            for prop in sample["type0"]["prop0"].tolist():
                assert prop % 31 == 0

        if "type1" in sample:
            for prop in sample["type1"]["prop0"].tolist():
                assert prop % 41 == 0


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_cugraph_loader_from_disk():
    F = FeatureStore()
    F.add_data(torch.tensor([1, 2, 3, 4, 5, 6, 7]), "t0", "x")

    G = {("t0", "knows", "t0"): 7}
    N = {"t0": 7}

    cugraph_store = CuGraphStore(F, G, N)

    bogus_samples = cudf.DataFrame(
        {
            "sources": [0, 1, 2, 3, 4, 5, 6],
            "destinations": [6, 4, 3, 2, 2, 1, 5],
            "edge_type": cudf.Series([0, 0, 0, 0, 0, 0, 0], dtype="int32"),
            "edge_id": [5, 10, 15, 20, 25, 30, 35],
            "hop_id": cudf.Series([0, 0, 0, 1, 1, 2, 2], dtype="int32"),
        }
    )

    tempdir = tempfile.TemporaryDirectory()
    for s in range(256):
        bogus_samples["batch_id"] = cupy.int32(s)
        bogus_samples.to_parquet(os.path.join(tempdir.name, f"batch={s}-{s}.parquet"))

    loader = BulkSampleLoader(
        feature_store=cugraph_store,
        graph_store=cugraph_store,
        directory=tempdir,
    )

    num_samples = 0
    for sample in loader:
        num_samples += 1
        assert sample["t0"]["num_nodes"] == 7
        # correct vertex order is [0, 1, 2, 6, 4, 3, 5]; x = [1, 2, 3, 7, 5, 4, 6]
        assert sample["t0"]["x"].tolist() == [1, 2, 3, 7, 5, 4, 6]
        assert list(sample[("t0", "knows", "t0")]["edge_index"].shape) == [2, 7]

    assert num_samples == 256


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_cugraph_loader_from_disk_subset():
    F = FeatureStore()
    F.add_data(torch.tensor([1, 2, 3, 4, 5, 6, 7]), "t0", "x")

    G = {("t0", "knows", "t0"): 7}
    N = {"t0": 7}

    cugraph_store = CuGraphStore(F, G, N)

    bogus_samples = cudf.DataFrame(
        {
            "sources": [0, 1, 2, 3, 4, 5, 6],
            "destinations": [6, 4, 3, 2, 2, 1, 5],
            "edge_type": cudf.Series([0, 0, 0, 0, 0, 0, 0], dtype="int32"),
            "edge_id": [5, 10, 15, 20, 25, 30, 35],
            "hop_id": cudf.Series([0, 0, 0, 1, 1, 2, 2], dtype="int32"),
        }
    )

    tempdir = tempfile.TemporaryDirectory()
    for s in range(256):
        bogus_samples["batch_id"] = cupy.int32(s)
        bogus_samples.to_parquet(os.path.join(tempdir.name, f"batch={s}-{s}.parquet"))

    loader = BulkSampleLoader(
        feature_store=cugraph_store,
        graph_store=cugraph_store,
        directory=tempdir,
        input_files=list(os.listdir(tempdir.name))[100:200],
    )

    num_samples = 0
    for sample in loader:
        num_samples += 1
        assert sample["t0"]["num_nodes"] == 7
        # correct vertex order is [0, 1, 2, 6, 4, 3, 5]; x = [1, 2, 3, 7, 5, 4, 6]
        assert sample["t0"]["x"].tolist() == [1, 2, 3, 7, 5, 4, 6]
        assert list(sample[("t0", "knows", "t0")]["edge_index"].shape) == [2, 7]

    assert num_samples == 100


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_cugraph_loader_from_disk_subset_renumbered():
    F = FeatureStore()
    F.add_data(torch.tensor([1, 2, 3, 4, 5, 6, 7]), "t0", "x")

    G = {("t0", "knows", "t0"): 7}
    N = {"t0": 7}

    cugraph_store = CuGraphStore(F, G, N)

    bogus_samples = cudf.DataFrame(
        {
            "sources": [0, 1, 2, 3, 4, 5, 6],
            "destinations": [6, 4, 3, 2, 2, 1, 5],
            "edge_type": cudf.Series([0, 0, 0, 0, 0, 0, 0], dtype="int32"),
            "edge_id": [5, 10, 15, 20, 25, 30, 35],
            "hop_id": cudf.Series([0, 0, 0, 1, 1, 2, 2], dtype="int32"),
        }
    )

    map = cudf.Series([2, 9, 0, 2, 1, 3, 4, 6, 5], name="map")
    bogus_samples = bogus_samples.join(map, how="outer").sort_index()

    tempdir = tempfile.TemporaryDirectory()
    for s in range(256):
        bogus_samples["batch_id"] = cupy.int32(s)
        bogus_samples.to_parquet(os.path.join(tempdir.name, f"batch={s}-{s}.parquet"))

    loader = BulkSampleLoader(
        feature_store=cugraph_store,
        graph_store=cugraph_store,
        directory=tempdir,
        input_files=list(os.listdir(tempdir.name))[100:200],
    )

    num_samples = 0
    for sample in loader:
        num_samples += 1
        assert sample["t0"]["num_nodes"] == 7
        # correct vertex order is [0, 2, 1, 3, 4, 6, 5]; x = [1, 3, 2, 4, 5, 7, 6]
        assert sample["t0"]["x"].tolist() == [1, 3, 2, 4, 5, 7, 6]

        edge_index = sample[("t0", "knows", "t0")]["edge_index"]
        assert list(edge_index.shape) == [2, 7]
        assert (
            edge_index[0].tolist()
            == bogus_samples.sources.dropna().values_host.tolist()
        )
        assert (
            edge_index[1].tolist()
            == bogus_samples.destinations.dropna().values_host.tolist()
        )

    assert num_samples == 100
