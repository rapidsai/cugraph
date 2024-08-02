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
