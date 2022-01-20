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

import pytest
import cupy as cp
import numpy as np


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


def test_sg_graph():
    from pylibcugraph.experimental import (SGGraph,
                                           ResourceHandle,
                                           GraphProperties,
    )

    graph_props = GraphProperties()
    graph_props.is_symmetric = False
    graph_props.is_multigraph = False
    resource_handle = ResourceHandle()

    srcs = cp.asarray([0, 1, 2], dtype=np.int32)
    dsts = cp.asarray([1, 2, 3], dtype=np.int32)
    weights = cp.asarray([0, 0, 0, 0], dtype=np.int32)

    g = SGGraph(resource_handle,
                graph_props,
                srcs,
                dsts,
                weights,
                store_transposed=False,
                renumber=False,
                expensive_check=False)
