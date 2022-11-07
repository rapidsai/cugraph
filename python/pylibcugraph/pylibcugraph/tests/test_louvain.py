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

import cupy as cp
import numpy as np
import cudf
from pylibcugraph import (
    SGGraph,
    ResourceHandle,
    GraphProperties,
)
from pylibcugraph import louvain


def check_results(d_vertices, d_clusters, modularity):
    expected_vertices = np.array([1, 2, 3, 0, 4, 5], dtype=np.int32)
    expected_clusters = np.array([0, 0, 0, 0, 1, 1], dtype=np.int32)
    expected_modularity = 0.125

    h_vertices = d_vertices.get()
    h_clusters = d_clusters.get()

    assert np.array_equal(expected_vertices, h_vertices)
    assert np.array_equal(expected_clusters, h_clusters)
    assert expected_modularity == modularity


# =============================================================================
# Pytest fixtures
# =============================================================================
# fixtures used in this test module are defined in conftest.py


# =============================================================================
# Tests
# =============================================================================
def test_sg_louvain_cupy():
    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=True, is_multigraph=False)

    device_srcs = cp.asarray(
        [0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5], dtype=np.int32
    )
    device_dsts = cp.asarray(
        [1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4], dtype=np.int32
    )
    device_weights = cp.asarray(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )

    max_level = 100
    resolution = 1.0

    sg = SGGraph(
        resource_handle,
        graph_props,
        device_srcs,
        device_dsts,
        device_weights,
        store_transposed=False,
        renumber=True,
        do_expensive_check=False,
    )

    vertices, clusters, modularity = louvain(
        resource_handle, sg, max_level, resolution, do_expensive_check=False
    )

    check_results(vertices, clusters, modularity)


def test_sg_louvain_cudf():
    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=True, is_multigraph=False)

    device_srcs = cudf.Series(
        [0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5], dtype=np.int32
    )
    device_dsts = cudf.Series(
        [1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4], dtype=np.int32
    )
    device_weights = cudf.Series(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )

    max_level = 100
    resolution = 1.0

    sg = SGGraph(
        resource_handle,
        graph_props,
        device_srcs,
        device_dsts,
        device_weights,
        store_transposed=False,
        renumber=True,
        do_expensive_check=False,
    )

    vertices, clusters, modularity = louvain(
        resource_handle, sg, max_level, resolution, do_expensive_check=False
    )

    check_results(vertices, clusters, modularity)
