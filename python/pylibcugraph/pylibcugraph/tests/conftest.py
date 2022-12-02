# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import pandas as pd
import cupy as cp
import numpy as np

import pytest

from pylibcugraph.testing import utils


# Spoof the gpubenchmark fixture if it's not available so that asvdb and
# rapids-pytest-benchmark do not need to be installed to run tests.
if "gpubenchmark" not in globals():

    def benchmark_func(func, *args, **kwargs):
        return func(*args, **kwargs)

    @pytest.fixture
    def gpubenchmark():
        return benchmark_func


# =============================================================================
# Fixture parameters
# =============================================================================
class COOTestGraphDeviceData:
    def __init__(self, srcs, dsts, weights, name):
        self.srcs = srcs
        self.dsts = dsts
        self.weights = weights
        self.name = name
        self.is_valid = not (name.startswith("Invalid"))


InvalidNumWeights_1 = COOTestGraphDeviceData(
    srcs=cp.asarray([0, 1, 2], dtype=np.int32),
    dsts=cp.asarray([1, 2, 3], dtype=np.int32),
    weights=cp.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    name="InvalidNumWeights_1",
)

InvalidNumVerts_1 = COOTestGraphDeviceData(
    srcs=cp.asarray([1, 2], dtype=np.int32),
    dsts=cp.asarray([1, 2, 3], dtype=np.int32),
    weights=cp.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    name="InvalidNumVerts_1",
)

Simple_1 = COOTestGraphDeviceData(
    srcs=cp.asarray([0, 1, 2], dtype=np.int32),
    dsts=cp.asarray([1, 2, 3], dtype=np.int32),
    weights=cp.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    name="Simple_1",
)

Simple_2 = COOTestGraphDeviceData(
    srcs=cp.asarray([0, 1, 1, 2, 2, 2, 3, 4], dtype=np.int32),
    dsts=cp.asarray([1, 3, 4, 0, 1, 3, 5, 5], dtype=np.int32),
    weights=cp.asarray([0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2], dtype=np.float32),
    name="Simple_2",
)


# The objects in these lists must have a "name" attr, since fixtures will
# access that to pass to tests, which then may use the name to associate to
# expected test results. The name attr is also used for the pytest test ID.
valid_datasets = [
    utils.RAPIDS_DATASET_ROOT_DIR_PATH / "karate.csv",
    utils.RAPIDS_DATASET_ROOT_DIR_PATH / "dolphins.csv",
    Simple_1,
    Simple_2,
]
all_datasets = valid_datasets + [
    InvalidNumWeights_1,
    InvalidNumVerts_1,
]


# =============================================================================
# Helper functions
# =============================================================================
def get_graph_data_for_dataset(ds, ds_name):
    """
    Given an object representing either a path to a dataset on disk, or an
    object containing raw data, return a series of arrays that can be used to
    construct a graph object. The final value is a bool used to indicate if the
    data is valid or not (invalid to test error handling).
    """
    if isinstance(ds, COOTestGraphDeviceData):
        device_srcs = ds.srcs
        device_dsts = ds.dsts
        device_weights = ds.weights
        is_valid = ds.is_valid
    else:
        pdf = pd.read_csv(
            ds,
            delimiter=" ",
            header=None,
            names=["0", "1", "weight"],
            dtype={"0": "int32", "1": "int32", "weight": "float32"},
        )
        device_srcs = cp.asarray(pdf["0"].to_numpy(), dtype=np.int32)
        device_dsts = cp.asarray(pdf["1"].to_numpy(), dtype=np.int32)
        device_weights = cp.asarray(pdf["weight"].to_numpy(), dtype=np.float32)
        # Assume all datasets on disk are valid
        is_valid = True

    return (device_srcs, device_dsts, device_weights, ds_name, is_valid)


def create_SGGraph(device_srcs, device_dsts, device_weights, transposed=False):
    """
    Creates and returns a SGGraph instance and the corresponding ResourceHandle
    using the parameters passed in.
    """
    from pylibcugraph import (
        SGGraph,
        ResourceHandle,
        GraphProperties,
    )

    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)

    g = SGGraph(
        resource_handle,
        graph_props,
        device_srcs,
        device_dsts,
        device_weights,
        store_transposed=transposed,
        renumber=False,
        do_expensive_check=False,
    )

    # FIXME: add coverage for renumber=True and do_expensive_check=True

    return (g, resource_handle)


# =============================================================================
# Pytest fixtures
# =============================================================================
@pytest.fixture(
    scope="package", params=[pytest.param(ds, id=ds.name) for ds in all_datasets]
)
def graph_data(request):
    """
    Return a series of cupy arrays that can be used to construct Graph
    objects. The parameterization includes invalid arrays which can be used to
    test error handling, so the final value returned indicated if the arrays
    are valid or not.
    """
    return get_graph_data_for_dataset(request.param, request.param.name)


@pytest.fixture(
    scope="package", params=[pytest.param(ds, id=ds.name) for ds in valid_datasets]
)
def valid_graph_data(request):
    """
    Return a series of cupy arrays that can be used to construct Graph objects,
    all of which are valid (last value in returned tuple is always True).
    """
    return get_graph_data_for_dataset(request.param, request.param.name)


@pytest.fixture(scope="package")
def sg_graph_objs(valid_graph_data, request):
    """
    Returns a tuple containing the SGGraph object constructed from
    parameterized values returned by the valid_graph_data fixture,
    the associated resource handle, and the name of the dataset
    used to construct the graph.
    """
    (device_srcs, device_dsts, device_weights, ds_name, is_valid) = valid_graph_data

    if is_valid is False:
        pytest.exit("got invalid graph data - expecting only valid data")

    (g, resource_handle) = create_SGGraph(
        device_srcs, device_dsts, device_weights, transposed=False
    )

    return (g, resource_handle, ds_name)


@pytest.fixture(scope="package")
def sg_transposed_graph_objs(valid_graph_data, request):
    """
    Returns a tuple containing the SGGraph object constructed from
    parameterized values returned by the valid_graph_data fixture,
    the associated resource handle, and the name of the dataset
    used to construct the graph.
    The SGGraph object is created with the transposed arg set to True.
    """
    (device_srcs, device_dsts, device_weights, ds_name, is_valid) = valid_graph_data

    if is_valid is False:
        pytest.exit("got invalid graph data - expecting only valid data")

    (g, resource_handle) = create_SGGraph(
        device_srcs, device_dsts, device_weights, transposed=True
    )

    return (g, resource_handle, ds_name)
