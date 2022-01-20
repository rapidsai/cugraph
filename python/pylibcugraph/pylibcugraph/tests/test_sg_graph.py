# Copyright (c) 2022, NVIDIA CORPORATION.
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

from . import utils


# =============================================================================
# Pytest fixtures
# =============================================================================
class InlineGraphData:
    @property
    def name(self):
        return self.__class__.__name__

    @property
    def is_valid(self):
        return not(self.name.startswith("Invalid"))

class InvalidNumWeights_1(InlineGraphData):  # noqa: E302
    srcs = cp.asarray([0, 1, 2], dtype=np.int32)
    dsts = cp.asarray([1, 2, 3], dtype=np.int32)
    weights = cp.asarray([0, 0, 0, 0], dtype=np.int32)

class InvalidNumVerts_1(InlineGraphData):  # noqa: E302
    srcs = cp.asarray([1, 2], dtype=np.int32)
    dsts = cp.asarray([1, 2, 3], dtype=np.int32)
    weights = cp.asarray([0, 0, 0], dtype=np.int32)

class Simple_1(InlineGraphData):  # noqa: E302
    srcs = cp.asarray([0, 1, 2], dtype=np.int32)
    dsts = cp.asarray([1, 2, 3], dtype=np.int32)
    weights = cp.asarray([0, 0, 0], dtype=np.int32)


datasets = [utils.RAPIDS_DATASET_ROOT_DIR_PATH/"karate.csv",
            utils.RAPIDS_DATASET_ROOT_DIR_PATH/"dolphins.csv",
            InvalidNumWeights_1(),
            InvalidNumVerts_1(),
            Simple_1(),
            ]


@pytest.fixture(scope="module",
                params=[pytest.param(ds, id=ds.name) for ds in datasets])
def graph_data(request):
    ds = request.param

    if isinstance(ds, InlineGraphData):
        device_srcs = ds.srcs
        device_dsts = ds.dsts
        device_weights = ds.weights
        is_valid = ds.is_valid
    else:
        pdf = pd.read_csv(ds,
                          delimiter=" ", header=None,
                          names=["0", "1", "weight"],
                          dtype={"0": "int32", "1": "int32",
                                 "weight": "float32"},
                          )
        device_srcs = cp.asarray(pdf["0"].to_numpy(), dtype=np.int32)
        device_dsts = cp.asarray(pdf["1"].to_numpy(), dtype=np.int32)
        device_weights = cp.asarray(pdf["weight"].to_numpy(), dtype=np.float32)
        # Assume all datasets on disk are valid
        is_valid = True

    return (device_srcs, device_dsts, device_weights, is_valid)


###############################################################################
# Tests
def test_graph_properties():
    from pylibcugraph.experimental import GraphProperties

    gp = GraphProperties()
    assert gp.is_symmetric is False
    assert gp.is_multigraph is False

    gp.is_symmetric = True
    assert gp.is_symmetric is True
    gp.is_symmetric = 0
    assert gp.is_symmetric is False
    with pytest.raises(TypeError):
        gp.is_symmetric = "foo"

    gp.is_multigraph = True
    assert gp.is_multigraph is True
    gp.is_multigraph = 0
    assert gp.is_multigraph is False
    with pytest.raises(TypeError):
        gp.is_multigraph = "foo"


def test_resource_handle():
    from pylibcugraph.experimental import ResourceHandle
    # This type has no attributes and is just defined to pass a struct from C
    # back in to C. In the future it may take args to acquire specific
    # resources, but for now just make sure nothing crashes.
    rh = ResourceHandle()
    del rh


def test_sg_graph_ctor(graph_data):

    from pylibcugraph.experimental import (SGGraph,
                                           ResourceHandle,
                                           GraphProperties,
                                           )

    (device_srcs, device_dsts, device_weights, is_valid) = graph_data

    graph_props = GraphProperties()
    graph_props.is_symmetric = False
    graph_props.is_multigraph = False
    resource_handle = ResourceHandle()

    if is_valid:
        g = SGGraph(resource_handle,
                    graph_props,
                    device_srcs,
                    device_dsts,
                    device_weights,
                    store_transposed=False,
                    renumber=False,
                    expensive_check=False)

        print(g)
    else:
        with pytest.raises(RuntimeError):
            g = SGGraph(resource_handle,
                        graph_props,
                        device_srcs,
                        device_dsts,
                        device_weights,
                        store_transposed=False,
                        renumber=False,
                        expensive_check=False)

            print(g)
