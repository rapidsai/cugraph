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
from cugraph_pyg.nn import SAGEConv as CuGraphSAGEConv

from cugraph.gnn import FeatureStore
from cugraph.utilities.utils import import_optional, MissingModule

torch = import_optional("torch")
torch_geometric = import_optional("torch_geometric")
trim_to_layer = import_optional("torch_geometric.utils.trim_to_layer")

try:
    import torch_sparse  # noqa: F401

    HAS_TORCH_SPARSE = True
except:  # noqa: E722
    HAS_TORCH_SPARSE = False


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
            "majors": [0, 1, 2, 3, 4, 5, 6, 6],
            "minors": [5, 4, 3, 2, 2, 6, 5, 2],
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
            edge_index[0].tolist() == bogus_samples.majors.dropna().values_host.tolist()
        )
        assert (
            edge_index[1].tolist() == bogus_samples.minors.dropna().values_host.tolist()
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
            "majors": [0, 1, 2, 3, 4, 5, 6, 6],
            "minors": [5, 4, 3, 2, 2, 6, 5, 2],
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
            edge_index[0].tolist() == bogus_samples.majors.dropna().values_host.tolist()
        )
        assert (
            edge_index[1].tolist() == bogus_samples.minors.dropna().values_host.tolist()
        )

    assert num_samples == 100


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(not HAS_TORCH_SPARSE, reason="torch-sparse not available")
def test_cugraph_loader_from_disk_subset_csr():
    m = [2, 9, 99, 82, 11, 13]
    n = torch.arange(1, 1 + len(m), dtype=torch.int32)
    x = torch.zeros(256, dtype=torch.int32)
    x[torch.tensor(m, dtype=torch.int32)] = n
    F = FeatureStore()
    F.add_data(x, "t0", "x")

    G = {("t0", "knows", "t0"): 9080}
    N = {"t0": 256}

    cugraph_store = CuGraphStore(F, G, N)

    bogus_samples = cudf.DataFrame(
        {
            "major_offsets": [0, 3, 5, 7, 8, None, None, None],
            "minors": [1, 2, 3, 0, 3, 4, 5, 1],
            "edge_type": cudf.Series([0, 0, 0, 0, 0, 0, 0, 0], dtype="int32"),
            "edge_id": [5, 10, 15, 20, 25, 30, 35, 40],
            "label_hop_offsets": cudf.Series(
                [0, 1, 4, None, None, None, None, None], dtype="int32"
            ),
            "renumber_map_offsets": cudf.Series([0, 6], dtype="int32"),
        }
    )
    map = cudf.Series(m, name="map")
    bogus_samples["map"] = map

    tempdir = tempfile.TemporaryDirectory()
    for s in range(256):
        # offset the offsets
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
        assert sample["t0"]["num_nodes"] == 6

        assert sample["t0"]["x"].tolist() == [1, 2, 3, 4, 5, 6]

        edge_index = sample[("t0", "knows", "t0")]["adj_t"]
        assert edge_index.size(0) == 4
        assert edge_index.size(1) == 6

        colptr, row, _ = edge_index.csr()

        assert (
            colptr.tolist() == bogus_samples.major_offsets.dropna().values_host.tolist()
        )
        assert row.tolist() == bogus_samples.minors.dropna().values_host.tolist()

        assert sample["t0"]["num_sampled_nodes"].tolist() == [1, 3, 2]
        assert sample["t0", "knows", "t0"]["num_sampled_edges"].tolist() == [3, 5]

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
            "majors": [0, 1, 2, 3, 4, 5, 6, 6],
            "minors": [5, 4, 3, 2, 2, 6, 5, 2],
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

        for i in range(len(convs)):
            x, ei, _ = trim(i, num_sampled_nodes, num_sampled_edges, x, ei, None)

            s = x.shape[0]

            x = convs[i](x, ei, size=(s, s))
            x = relu(x)
            x = dropout(x, p=0.5)

        x = x.narrow(dim=0, start=0, length=x.shape[0] - num_sampled_nodes[1])

        assert list(x.shape) == [3, 1]


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(not HAS_TORCH_SPARSE, reason="torch-sparse not available")
@pytest.mark.parametrize("framework", ["pyg", "cugraph-ops"])
def test_cugraph_loader_e2e_csc(framework):
    m = [2, 9, 99, 82, 9, 3, 18, 1, 12]
    x = torch.randint(3000, (256, 256)).to(torch.float32)
    F = FeatureStore()
    F.add_data(x, "t0", "x")

    G = {("t0", "knows", "t0"): 9999}
    N = {"t0": 256}

    cugraph_store = CuGraphStore(F, G, N)

    bogus_samples = cudf.DataFrame(
        {
            "major_offsets": [0, 3, 5, 7, 8, None, None, None],
            "minors": [1, 2, 3, 0, 3, 4, 5, 1],
            "edge_type": cudf.Series([0, 0, 0, 0, 0, 0, 0, 0], dtype="int32"),
            "edge_id": [5, 10, 15, 20, 25, 30, 35, 40],
            "label_hop_offsets": cudf.Series(
                [0, 1, 4, None, None, None, None, None], dtype="int32"
            ),
            "renumber_map_offsets": cudf.Series([0, 6], dtype="int32"),
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

    if framework == "pyg":
        convs = [
            torch_geometric.nn.SAGEConv(256, 64, aggr="mean").cuda(),
            torch_geometric.nn.SAGEConv(64, 1, aggr="mean").cuda(),
        ]
    else:
        convs = [
            CuGraphSAGEConv(256, 64, aggr="mean").cuda(),
            CuGraphSAGEConv(64, 1, aggr="mean").cuda(),
        ]

    trim = trim_to_layer.TrimToLayer()
    relu = torch.nn.functional.relu
    dropout = torch.nn.functional.dropout

    for hetero_data in loader:
        x = hetero_data["t0"]["x"].cuda()

        if framework == "pyg":
            ei = hetero_data["t0", "knows", "t0"]["adj_t"].coo()
            ei = torch.stack((ei[0], ei[1]))
        else:
            ei = hetero_data["t0", "knows", "t0"]["adj_t"].csr()
            ei = [ei[1], ei[0], x.shape[0]]

        num_sampled_nodes = hetero_data["t0"]["num_sampled_nodes"]
        num_sampled_edges = hetero_data["t0", "knows", "t0"]["num_sampled_edges"]

        s = x.shape[0]
        for i in range(len(convs)):
            if framework == "pyg":
                x, ei, _ = trim(i, num_sampled_nodes, num_sampled_edges, x, ei, None)
            else:
                if i > 0:
                    x = x.narrow(
                        dim=0,
                        start=0,
                        length=s - num_sampled_nodes[-i],
                    )

                    ei[0] = ei[0].narrow(
                        dim=0,
                        start=0,
                        length=ei[0].size(0) - num_sampled_edges[-i],
                    )
                    ei[1] = ei[1].narrow(
                        dim=0, start=0, length=ei[1].size(0) - num_sampled_nodes[-i]
                    )
                    ei[2] = x.size(0)

            s = x.shape[0]

            if framework == "pyg":
                x = convs[i](x, ei, size=(s, s))
            else:
                x = convs[i](x, ei)
            x = relu(x)
            x = dropout(x, p=0.5)

        x = x.narrow(dim=0, start=0, length=s - num_sampled_nodes[1])

        assert list(x.shape) == [1, 1]
