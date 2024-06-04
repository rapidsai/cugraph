# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

from cugraph_pyg.loader import DaskNeighborLoader
from cugraph_pyg.data import DaskGraphStore
from cugraph.utilities.utils import import_optional, MissingModule

torch = import_optional("torch")


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.mg
def test_cugraph_loader_basic(dask_client, karate_gnn):
    F, G, N = karate_gnn
    cugraph_store = DaskGraphStore(F, G, N, multi_gpu=True, order="CSR")
    loader = DaskNeighborLoader(
        (cugraph_store, cugraph_store),
        torch.arange(N["type0"] + N["type1"], dtype=torch.int64),
        10,
        num_neighbors=[4, 4],
        random_state=62,
        replace=False,
    )

    assert isinstance(cugraph_store._subgraph()._plc_graph, dict)

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
@pytest.mark.mg
def test_cugraph_loader_hetero(dask_client, karate_gnn):
    F, G, N = karate_gnn
    cugraph_store = DaskGraphStore(F, G, N, multi_gpu=True, order="CSR")
    loader = DaskNeighborLoader(
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
        print(sample)
        if "type0" in sample:
            for prop in sample["type0"]["prop0"].tolist():
                assert prop % 31 == 0

        if "type1" in sample:
            for prop in sample["type1"]["prop0"].tolist():
                assert prop % 41 == 0
