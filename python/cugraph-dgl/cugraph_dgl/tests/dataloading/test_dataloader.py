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

import cugraph_dgl
from cugraph_dgl.dataloading.dataloader import DataLoader
from cugraph_dgl.dataloading import NeighborSampler

from cugraph.datasets import karate
from cugraph.utilities.utils import import_optional, MissingModule

torch = import_optional('torch')
dgl = import_optional('dgl')

@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(isinstance(dgl, MissingModule), reason="dgl not available")
def test_dataloader_basic_homogeneous():
    graph = cugraph_dgl.Graph(
        is_multi_gpu=False
    )

    num_nodes = karate.number_of_nodes()
    graph.add_nodes(
        num_nodes,
        data={'z': torch.arange(num_nodes)}
    )

    edf = karate.get_edgelist()
    graph.add_edges(
        u=edf['src'],
        v=edf['dst'],
        data={'q': torch.arange(karate.number_of_edges())}
    )

    sampler = NeighborSampler([5, 5, 5])
    loader = DataLoader(graph, torch.arange(num_nodes), sampler, batch_size=2)

    print(next(iter(loader)))