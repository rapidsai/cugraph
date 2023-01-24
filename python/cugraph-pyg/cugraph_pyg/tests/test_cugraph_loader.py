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

import numpy as np
import cupy
import cudf

from cugraph.experimental.datasets import karate
from cugraph.gnn import FeatureStore
from cugraph_pyg.loader import CuGraphNeighborLoader
from cugraph_pyg.data import CuGraphStore

import pytest


@pytest.fixture
def karate_gnn():
    el = karate.get_edgelist().reset_index(drop=True)
    el.src = el.src.astype("int64")
    el.dst = el.dst.astype("int64")
    all_vertices = np.array_split(cudf.concat([el.src, el.dst]).unique().values_host, 2)

    F = FeatureStore(backend="numpy")
    F.add_data(
        np.arange(len(all_vertices[0]), dtype="float32") * 31,
        "type0",
        "prop0",
    )
    F.add_data(
        np.arange(len(all_vertices[1]), dtype="float32") * 41,
        "type1",
        "prop0",
    )

    N = {
        "type0": len(all_vertices[0]),
        "type1": len(all_vertices[1]),
    }

    offsets = {"type0": 0, "type1": N["type0"]}

    G = {
        ("type0", "et01", "type1"): el[
            el.src.isin(all_vertices[0]) & el.dst.isin(all_vertices[1])
        ],
        ("type1", "et10", "type0"): el[
            el.src.isin(all_vertices[1]) & el.dst.isin(all_vertices[0])
        ],
        ("type0", "et00", "type0"): el[
            el.src.isin(all_vertices[0]) & el.dst.isin(all_vertices[0])
        ],
        ("type1", "et11", "type1"): el[
            el.src.isin(all_vertices[1]) & el.dst.isin(all_vertices[1])
        ],
    }

    G = {
        (src_type, edge_type, dst_type): (
            elx["src"].values_host - offsets[src_type],
            elx["dst"].values_host - offsets[dst_type],
        )
        for (src_type, edge_type, dst_type), elx in G.items()
    }

    return F, G, N


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
    print(samples)

    assert len(samples) == 3
    for sample in samples:
        for prop in sample["type0"]["prop0"].tolist():
            assert prop % 31 == 0
        for prop in sample["type1"]["prop0"].tolist():
            assert prop % 41 == 0
