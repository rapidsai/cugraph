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
torch_geometric = import_optional("torch_geometric")
trim_to_layer = import_optional("torch_geometric.utils.trim_to_layer")


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_cugraph_loader_basic(karate_gnn):
    F, G, N = karate_gnn
    cugraph_store = CuGraphStore(F, G, N, order="CSR")
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
    cugraph_store = CuGraphStore(F, G, N, order="CSR")
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
    m = [2, 9, 99, 82, 9, 3, 18, 1, 12]
    n = torch.arange(1, 1 + len(m), dtype=torch.int32)
    x = torch.zeros(256, dtype=torch.int32)
    x[torch.tensor(m, dtype=torch.int32)] = n
    F = FeatureStore()
    F.add_data(x, "t0", "x")

    G = {("t0", "knows", "t0"): 9080}
    N = {"t0": 256}

    cugraph_store = CuGraphStore(F, G, N, order="CSR")

    bogus_samples = cudf.DataFrame(
        {
            "sources": [0, 1, 2, 3, 4, 5, 6, 6],
            "destinations": [5, 4, 3, 2, 2, 6, 5, 2],
            "edge_type": cudf.Series([0, 0, 0, 0, 0, 0, 0, 0], dtype="int32"),
            "edge_id": [5, 10, 15, 20, 25, 30, 35, 40],
            "hop_id": cudf.Series([0, 0, 0, 1, 1, 1, 2, 2], dtype="int32"),
        }
    )
    map = cudf.Series(m, name="map")
    bogus_samples = bogus_samples.join(map, how="outer").sort_index()

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
        # correct vertex order is [0, 1, 2, 5, 4, 3, 6]; x = [1, 2, 3, 6, 5, 4, 7]
        assert sample["t0"]["x"].tolist() == [3, 4, 5, 6, 7, 8, 9]

        edge_index = sample[("t0", "knows", "t0")]["edge_index"]
        assert list(edge_index.shape) == [2, 8]

        assert (
            edge_index[0].tolist()
            == bogus_samples.sources.dropna().values_host.tolist()
        )
        assert (
            edge_index[1].tolist()
            == bogus_samples.destinations.dropna().values_host.tolist()
        )

    assert num_samples == 256


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_cugraph_loader_from_disk_subset():
    m = [2, 9, 99, 82, 9, 3, 18, 1, 12]
    n = torch.arange(1, 1 + len(m), dtype=torch.int32)
    x = torch.zeros(256, dtype=torch.int32)
    x[torch.tensor(m, dtype=torch.int32)] = n
    F = FeatureStore()
    F.add_data(x, "t0", "x")

    G = {("t0", "knows", "t0"): 9080}
    N = {"t0": 256}

    cugraph_store = CuGraphStore(F, G, N, order="CSR")

    bogus_samples = cudf.DataFrame(
        {
            "sources": [0, 1, 2, 3, 4, 5, 6, 6],
            "destinations": [5, 4, 3, 2, 2, 6, 5, 2],
            "edge_type": cudf.Series([0, 0, 0, 0, 0, 0, 0, 0], dtype="int32"),
            "edge_id": [5, 10, 15, 20, 25, 30, 35, 40],
            "hop_id": cudf.Series([0, 0, 0, 1, 1, 1, 2, 2], dtype="int32"),
        }
    )
    map = cudf.Series(m, name="map")
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
        # correct vertex order is [0, 1, 2, 6, 4, 3, 5]; x = [1, 2, 3, 7, 5, 4, 6]
        assert sample["t0"]["x"].tolist() == [3, 4, 5, 6, 7, 8, 9]

        edge_index = sample[("t0", "knows", "t0")]["edge_index"]
        assert list(edge_index.shape) == [2, 8]

        assert (
            edge_index[0].tolist()
            == bogus_samples.sources.dropna().values_host.tolist()
        )
        assert (
            edge_index[1].tolist()
            == bogus_samples.destinations.dropna().values_host.tolist()
        )

    assert num_samples == 100


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_cugraph_loader_e2e_coo():
    m = [2, 9, 99, 82, 9, 3, 18, 1, 12]
    x = torch.randint(3000, (256, 256)).to(torch.float32)
    F = FeatureStore()
    F.add_data(x, "t0", "x")

    G = {("t0", "knows", "t0"): 9999}
    N = {"t0": 256}

    cugraph_store = CuGraphStore(F, G, N, order="CSR")

    bogus_samples = cudf.DataFrame(
        {
            "sources": [0, 1, 2, 3, 4, 5, 6, 6],
            "destinations": [5, 4, 3, 2, 2, 6, 5, 2],
            "edge_type": cudf.Series([0, 0, 0, 0, 0, 0, 0, 0], dtype="int32"),
            "edge_id": [5, 10, 15, 20, 25, 30, 35, 40],
            "hop_id": cudf.Series([0, 0, 0, 1, 1, 1, 2, 2], dtype="int32"),
        }
    )
    map = cudf.Series(m, name="map")
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

    convs = [
        torch_geometric.nn.SAGEConv(256, 64, aggr="mean").cuda(),
        torch_geometric.nn.SAGEConv(64, 8, aggr="mean").cuda(),
        torch_geometric.nn.SAGEConv(8, 1, aggr="mean").cuda(),
    ]

    trim = trim_to_layer.TrimToLayer()
    relu = torch.nn.functional.relu
    dropout = torch.nn.functional.dropout

    for hetero_data in loader:
        ei = hetero_data["t0", "knows", "t0"]["edge_index"]
        x = hetero_data["t0"]["x"].cuda()
        num_sampled_nodes = hetero_data["t0"]["num_sampled_nodes"]
        num_sampled_edges = hetero_data["t0", "knows", "t0"]["num_sampled_edges"]

        print(num_sampled_nodes, num_sampled_edges)

        for i in range(len(convs)):
            x, ei, _ = trim(i, num_sampled_nodes, num_sampled_edges, x, ei, None)

            s = x.shape[0]

            x = convs[i](x, ei, size=(s, s))
            x = relu(x)
            x = dropout(x, p=0.5)
            print(x.shape)

        print(x.shape)
        x = x.narrow(dim=0, start=0, length=x.shape[0] - num_sampled_nodes[1])

        assert list(x.shape) == [3, 1]
