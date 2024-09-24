# Copyright (c) 2024, NVIDIA CORPORATION.
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

from cugraph.datasets import karate
from cugraph.utilities.utils import import_optional, MissingModule

import cugraph_pyg
from cugraph_pyg.data import TensorDictFeatureStore, GraphStore
from cugraph_pyg.loader import NeighborLoader

torch = import_optional("torch")
torch_geometric = import_optional("torch_geometric")


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_neighbor_loader():
    """
    Basic e2e test that covers loading and sampling.
    """

    df = karate.get_edgelist()
    src = torch.as_tensor(df["src"], device="cuda")
    dst = torch.as_tensor(df["dst"], device="cuda")

    ei = torch.stack([dst, src])

    graph_store = GraphStore()
    graph_store.put_edge_index(ei, ("person", "knows", "person"), "coo")

    feature_store = TensorDictFeatureStore()
    feature_store["person", "feat"] = torch.randint(128, (34, 16))

    loader = NeighborLoader(
        (feature_store, graph_store),
        [5, 5],
        input_nodes=torch.arange(34),
    )

    for batch in loader:
        assert isinstance(batch, torch_geometric.data.Data)
        assert (feature_store["person", "feat"][batch.n_id] == batch.feat).all()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_neighbor_loader_biased():
    eix = torch.tensor(
        [
            [3, 4, 5],
            [0, 1, 2],
        ]
    )

    graph_store = GraphStore()
    graph_store.put_edge_index(eix, ("person", "knows", "person"), "coo")

    feature_store = TensorDictFeatureStore()
    feature_store["person", "feat"] = torch.randint(128, (6, 12))
    feature_store[("person", "knows", "person"), "bias"] = torch.tensor(
        [0, 12, 14], dtype=torch.float32
    )

    loader = NeighborLoader(
        (feature_store, graph_store),
        [1],
        input_nodes=torch.tensor([0, 1, 2], dtype=torch.int64),
        batch_size=3,
        weight_attr="bias",
    )

    out = list(iter(loader))
    assert len(out) == 1
    out = out[0]

    assert out.edge_index.shape[1] == 2
    assert (out.edge_index.cpu() == torch.tensor([[3, 4], [1, 2]])).all()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
@pytest.mark.parametrize("num_nodes", [10, 25])
@pytest.mark.parametrize("num_edges", [64, 128])
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("select_edges", [16, 32])
@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("num_neighbors", [1, 4])
def test_link_neighbor_loader_basic(
    num_nodes, num_edges, batch_size, select_edges, num_neighbors, depth
):
    graph_store = GraphStore()
    feature_store = TensorDictFeatureStore()

    eix = torch.randperm(num_edges)[:select_edges]
    graph_store[("n", "e", "n"), "coo"] = torch.stack(
        [
            torch.randint(0, num_nodes, (num_edges,)),
            torch.randint(0, num_nodes, (num_edges,)),
        ]
    )

    elx = graph_store[("n", "e", "n"), "coo"][:, eix]
    loader = cugraph_pyg.loader.LinkNeighborLoader(
        (feature_store, graph_store),
        num_neighbors=[num_neighbors] * depth,
        edge_label_index=elx,
        batch_size=batch_size,
        shuffle=False,
    )

    elx = torch.tensor_split(elx, eix.numel() // batch_size, dim=1)
    for i, batch in enumerate(loader):
        assert (
            batch.input_id.cpu() == torch.arange(i * batch_size, (i + 1) * batch_size)
        ).all()
        assert (elx[i] == batch.n_id[batch.edge_label_index.cpu()]).all()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
@pytest.mark.parametrize("batch_size", [1, 2])
def test_link_neighbor_loader_negative_sampling(batch_size):
    num_edges = 62
    num_nodes = 19
    select_edges = 17

    graph_store = GraphStore()
    feature_store = TensorDictFeatureStore()

    eix = torch.randperm(num_edges)[:select_edges]
    graph_store[("n", "e", "n"), "coo"] = torch.stack(
        [
            torch.randint(0, num_nodes, (num_edges,)),
            torch.randint(0, num_nodes, (num_edges,)),
        ]
    )

    elx = graph_store[("n", "e", "n"), "coo"][:, eix]
    loader = cugraph_pyg.loader.LinkNeighborLoader(
        (feature_store, graph_store),
        num_neighbors=[3, 3, 3],
        edge_label_index=elx,
        batch_size=batch_size,
        shuffle=False,
    )

    elx = torch.tensor_split(elx, eix.numel() // batch_size, dim=1)
    for i, batch in enumerate(loader):
        assert (
            batch.input_id.cpu() == torch.arange(i * batch_size, (i + 1) * batch_size)
        ).all()
        assert (elx[i] == batch.n_id[batch.edge_label_index.cpu()]).all()
