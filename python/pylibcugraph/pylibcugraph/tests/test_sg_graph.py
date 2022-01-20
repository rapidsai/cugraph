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
datasets = [utils.RAPIDS_DATASET_ROOT_DIR_PATH/"karate.csv",
            utils.RAPIDS_DATASET_ROOT_DIR_PATH/"dolphins.csv",
            ]


@pytest.fixture(scope="module",
                params=[pytest.param(ds, id=ds.name) for ds in datasets])
def graph_arrays(request):
    ds = request.param

    pdf = pd.read_csv(ds,
                      delimiter=" ", header=None,
                      names=["0", "1", "weight"],
                      dtype={"0": "int32", "1": "int32", "weight": "float32"},
                      )
    device_srcs = cp.asarray(pdf["0"].to_numpy(), dtype=np.int32)
    device_dsts = cp.asarray(pdf["1"].to_numpy(), dtype=np.int32)
    device_weights = cp.asarray(pdf["weight"].to_numpy(), dtype=np.float32)

    return (device_srcs, device_dsts, device_weights)


###############################################################################
# Tests
def test_ctor(graph_arrays):
    from pylibcugraph.experimental import SGGraph

    (device_srcs, device_dsts, device_weights) = graph_arrays

    G = SGGraph(src_array=device_srcs,
                dst_array=device_dsts,
                weight_array=device_weights,
                store_transposed=False)

    print(G)
    # FIXME: test for correct num verts, edges, etc.
