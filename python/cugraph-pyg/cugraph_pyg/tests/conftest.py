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

import numpy as np
import cudf
from cugraph.gnn import FeatureStore
from cugraph.experimental.datasets import karate

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


@pytest.fixture
def basic_graph_1():
    G = {
        ("vt1", "pig", "vt1"): [
            np.array([0, 0, 1, 2, 2, 3]),
            np.array([1, 2, 4, 3, 4, 1]),
        ]
    }

    N = {"vt1": 5}

    F = FeatureStore()
    F.add_data(np.array([100, 200, 300, 400, 500]), type_name="vt1", feat_name="prop1")

    F.add_data(np.array([5, 4, 3, 2, 1]), type_name="vt1", feat_name="prop2")

    return F, G, N


@pytest.fixture
def multi_edge_graph_1():
    G = {
        ("vt1", "pig", "vt1"): [np.array([0, 2, 3, 1]), np.array([1, 3, 1, 4])],
        ("vt1", "dog", "vt1"): [np.array([0, 3, 4]), np.array([2, 2, 3])],
        ("vt1", "cat", "vt1"): [
            np.array([1, 2, 2]),
            np.array([4, 3, 4]),
        ],
    }

    N = {"vt1": 5}

    F = FeatureStore()
    F.add_data(np.array([100, 200, 300, 400, 500]), type_name="vt1", feat_name="prop1")

    F.add_data(np.array([5, 4, 3, 2, 1]), type_name="vt1", feat_name="prop2")

    return F, G, N


@pytest.fixture
def multi_edge_multi_vertex_graph_1():

    G = {
        ("brown", "horse", "brown"): [
            np.array([0, 0]),
            np.array([1, 2]),
        ],
        ("brown", "tortoise", "black"): [
            np.array([1, 1, 2]),
            np.array([1, 0, 1]),
        ],
        ("brown", "mongoose", "black"): [
            np.array([2, 1]),
            np.array([0, 1]),
        ],
        ("black", "cow", "brown"): [
            np.array([0, 0]),
            np.array([1, 2]),
        ],
        ("black", "snake", "black"): [
            np.array([1]),
            np.array([0]),
        ],
    }

    N = {"brown": 3, "black": 2}

    F = FeatureStore()
    F.add_data(np.array([100, 200, 300]), type_name="brown", feat_name="prop1")

    F.add_data(np.array([400, 500]), type_name="black", feat_name="prop1")

    F.add_data(np.array([5, 4, 3]), type_name="brown", feat_name="prop2")

    F.add_data(np.array([2, 1]), type_name="black", feat_name="prop2")

    return F, G, N
