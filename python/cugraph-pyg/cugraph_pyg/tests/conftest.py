# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

import os
import pytest

from dask_cuda.initialize import initialize as dask_initialize
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from cugraph.dask.comms import comms as Comms
from cugraph.dask.common.mg_utils import get_visible_devices
from cugraph.testing.mg_utils import stop_dask_client

import torch
import numpy as np
from cugraph.gnn import FeatureStore
from cugraph.datasets import karate

import tempfile

# module-wide fixtures

# If the rapids-pytest-benchmark plugin is installed, the "gpubenchmark"
# fixture will be available automatically. Check that this fixture is available
# by trying to import rapids_pytest_benchmark, and if that fails, set
# "gpubenchmark" to the standard "benchmark" fixture provided by
# pytest-benchmark.
try:
    import rapids_pytest_benchmark  # noqa: F401
except ImportError:
    import pytest_benchmark

    gpubenchmark = pytest_benchmark.plugin.benchmark


@pytest.fixture(scope="module")
def dask_client():
    dask_scheduler_file = os.environ.get("SCHEDULER_FILE")
    cuda_visible_devices = get_visible_devices()

    if dask_scheduler_file is not None:
        dask_initialize()
        dask_client = Client(scheduler_file=dask_scheduler_file)
    else:
        # The tempdir created by tempdir_object should be cleaned up once
        # tempdir_object goes out-of-scope and is deleted.
        tempdir_object = tempfile.TemporaryDirectory()
        cluster = LocalCUDACluster(
            local_directory=tempdir_object.name,
            protocol="tcp",
            CUDA_VISIBLE_DEVICES=cuda_visible_devices,
        )

        dask_client = Client(cluster)
        dask_client.wait_for_workers(len(cuda_visible_devices))

    if not Comms.is_initialized():
        Comms.initialize(p2p=True)

    yield dask_client

    stop_dask_client(dask_client)
    print("\ndask_client fixture: client.close() called")


@pytest.fixture
def karate_gnn():
    el = karate.get_edgelist().reset_index(drop=True)
    el.src = el.src.astype("int64")
    el.dst = el.dst.astype("int64")
    all_vertices = np.array_split(np.arange(34), 2)

    F = FeatureStore(backend="torch")
    F.add_data(
        torch.arange(len(all_vertices[0]), dtype=torch.float32) * 31,
        "type0",
        "prop0",
    )
    F.add_data(
        torch.arange(len(all_vertices[1]), dtype=torch.float32) * 41,
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
        ].reset_index(drop=True),
        ("type1", "et10", "type0"): el[
            el.src.isin(all_vertices[1]) & el.dst.isin(all_vertices[0])
        ].reset_index(drop=True),
        ("type0", "et00", "type0"): el[
            el.src.isin(all_vertices[0]) & el.dst.isin(all_vertices[0])
        ],
        ("type1", "et11", "type1"): el[
            el.src.isin(all_vertices[1]) & el.dst.isin(all_vertices[1])
        ].reset_index(drop=True),
    }

    G = {
        (src_type, edge_type, dst_type): (
            torch.tensor(elx["src"].values_host - offsets[src_type]),
            torch.tensor(elx["dst"].values_host - offsets[dst_type]),
        )
        for (src_type, edge_type, dst_type), elx in G.items()
    }

    return F, G, N


@pytest.fixture
def basic_graph_1():
    G = {
        ("vt1", "pig", "vt1"): [
            torch.tensor([0, 0, 1, 2, 2, 3]),
            torch.tensor([1, 2, 4, 3, 4, 1]),
        ]
    }

    N = {"vt1": 5}

    F = FeatureStore()
    F.add_data(
        torch.tensor([100, 200, 300, 400, 500]), type_name="vt1", feat_name="prop1"
    )

    F.add_data(torch.tensor([5, 4, 3, 2, 1]), type_name="vt1", feat_name="prop2")

    return F, G, N


@pytest.fixture
def multi_edge_graph_1():
    G = {
        ("vt1", "pig", "vt1"): [torch.tensor([0, 2, 3, 1]), torch.tensor([1, 3, 1, 4])],
        ("vt1", "dog", "vt1"): [torch.tensor([0, 3, 4]), torch.tensor([2, 2, 3])],
        ("vt1", "cat", "vt1"): [
            torch.tensor([1, 2, 2]),
            torch.tensor([4, 3, 4]),
        ],
    }

    N = {"vt1": 5}

    F = FeatureStore()
    F.add_data(
        torch.tensor([100, 200, 300, 400, 500]), type_name="vt1", feat_name="prop1"
    )

    F.add_data(torch.tensor([5, 4, 3, 2, 1]), type_name="vt1", feat_name="prop2")

    return F, G, N


@pytest.fixture
def multi_edge_multi_vertex_graph_1():

    G = {
        ("brown", "horse", "brown"): [
            torch.tensor([0, 0]),
            torch.tensor([1, 2]),
        ],
        ("brown", "tortoise", "black"): [
            torch.tensor([1, 1, 2]),
            torch.tensor([1, 0, 1]),
        ],
        ("brown", "mongoose", "black"): [
            torch.tensor([2, 1]),
            torch.tensor([0, 1]),
        ],
        ("black", "cow", "brown"): [
            torch.tensor([0, 0]),
            torch.tensor([1, 2]),
        ],
        ("black", "snake", "black"): [
            torch.tensor([1]),
            torch.tensor([0]),
        ],
    }

    N = {"brown": 3, "black": 2}

    F = FeatureStore()
    F.add_data(torch.tensor([100, 200, 300]), type_name="brown", feat_name="prop1")

    F.add_data(torch.tensor([400, 500]), type_name="black", feat_name="prop1")

    F.add_data(torch.tensor([5, 4, 3]), type_name="brown", feat_name="prop2")

    F.add_data(torch.tensor([2, 1]), type_name="black", feat_name="prop2")

    return F, G, N


@pytest.fixture
def multi_edge_multi_vertex_no_graph_1():
    G = {
        ("brown", "horse", "brown"): 2,
        ("brown", "tortoise", "black"): 3,
        ("brown", "mongoose", "black"): 3,
        ("black", "cow", "brown"): 3,
        ("black", "snake", "black"): 1,
    }

    N = {"brown": 3, "black": 2}

    F = FeatureStore()
    F.add_data(np.array([100, 200, 300]), type_name="brown", feat_name="prop1")

    F.add_data(np.array([400, 500]), type_name="black", feat_name="prop1")

    F.add_data(np.array([5, 4, 3]), type_name="brown", feat_name="prop2")

    F.add_data(np.array([2, 1]), type_name="black", feat_name="prop2")

    return F, G, N


@pytest.fixture
def abc_graph():
    N = {
        "A": 2,  # 0, 1
        "B": 3,  # 2, 3, 4
        "C": 4,  # 5, 6, 7, 8
    }

    G = {
        # (0->2, 0->3, 1->3)
        ("A", "ab", "B"): [
            torch.tensor([0, 0, 1], dtype=torch.int64),
            torch.tensor([0, 1, 1], dtype=torch.int64),
        ],
        # (2->0, 2->1, 3->1, 4->0)
        ("B", "ba", "A"): [
            torch.tensor([0, 0, 1, 2], dtype=torch.int64),
            torch.tensor([0, 1, 1, 0], dtype=torch.int64),
        ],
        # (2->6, 2->8, 3->5, 3->7, 4->5, 4->8)
        ("B", "bc", "C"): [
            torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),
            torch.tensor([1, 3, 0, 2, 0, 3], dtype=torch.int64),
        ],
    }

    F = FeatureStore()
    F.add_data(
        torch.tensor([3.2, 2.1], dtype=torch.float32), type_name="A", feat_name="prop1"
    )

    return F, G, N


@pytest.fixture
def basic_pyg_graph_1():
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    size = (4, 4)
    return edge_index, size


@pytest.fixture
def basic_pyg_graph_2():
    edge_index = torch.tensor(
        [
            [0, 1, 0, 2, 3, 0, 4, 0, 5, 0, 6, 7, 0, 8, 9],
            [1, 9, 2, 9, 9, 4, 9, 5, 9, 6, 9, 9, 8, 9, 0],
        ]
    )
    size = (10, 10)
    return edge_index, size


@pytest.fixture
def sample_pyg_hetero_data():
    torch.manual_seed(12345)
    raw_data_dict = {
        "v0": torch.randn(6, 3),
        "v1": torch.randn(7, 2),
        "v2": torch.randn(5, 4),
        ("v2", "e0", "v1"): torch.tensor([[0, 2, 2, 4, 4], [4, 3, 6, 0, 1]]),
        ("v1", "e1", "v1"): torch.tensor(
            [[0, 2, 2, 2, 3, 5, 5], [4, 0, 4, 5, 3, 0, 1]]
        ),
        ("v0", "e2", "v0"): torch.tensor([[0, 2, 2, 3, 5, 5], [1, 1, 5, 1, 1, 2]]),
        ("v1", "e3", "v2"): torch.tensor(
            [[0, 1, 1, 2, 4, 5, 6], [1, 2, 3, 1, 2, 2, 2]]
        ),
        ("v0", "e4", "v2"): torch.tensor([[1, 1, 3, 3, 4, 4], [1, 4, 1, 4, 0, 3]]),
    }

    # create a nested dictionary to facilitate PyG's HeteroData construction
    hetero_data_dict = {}
    for key, value in raw_data_dict.items():
        if isinstance(key, tuple):
            hetero_data_dict[key] = {"edge_index": value}
        else:
            hetero_data_dict[key] = {"x": value}

    return hetero_data_dict
