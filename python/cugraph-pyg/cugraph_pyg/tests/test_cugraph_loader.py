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

import cupy

from cugraph_pyg.loader import CuGraphNeighborLoader
from cugraph_pyg.data import CuGraphStore


def test_cugraph_loader_basic(karate_gnn):
    F, G, N = karate_gnn
    cugraph_store = CuGraphStore(F, G, N, backend="cupy")
    loader = CuGraphNeighborLoader(
        (cugraph_store, cugraph_store),
        cupy.arange(N["type0"] + N["type1"], dtype="int64"),
        10,
        num_neighbors=[4, 4],
        random_state=62,
        replace=False,
    )

    samples = [s for s in loader]

    assert len(samples) == 3
    for sample in samples:
        for prop in sample["type0"]["prop0"].tolist():
            assert prop % 31 == 0
        for prop in sample["type1"]["prop0"].tolist():
            assert prop % 41 == 0
