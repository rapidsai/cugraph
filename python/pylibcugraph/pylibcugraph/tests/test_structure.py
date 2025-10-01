# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
from pylibcugraph import (
    SGGraph,
    ResourceHandle,
    GraphProperties,
)


# =============================================================================
# Tests
# =============================================================================
def test_type_combinations():
    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=True, is_multigraph=False)

    device_srcs = cp.asarray(
        [0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5], dtype=np.int32
    )
    device_dsts = cp.asarray(
        [1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4], dtype=np.int32
    )
    # Invalid combination of vertices and edgelist majors and minor
    device_vertices = cp.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
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

    warning_msg = (
        "The graph requires 'src_or_offset_array', 'dst_or_index_array' "
        "'vertices_array' and 'edge_id_array' to match. "
        "Those will be widened to 64-bit."
    )

    with pytest.warns(UserWarning, match=warning_msg):
        SGGraph(
            resource_handle=resource_handle,
            graph_properties=graph_props,
            src_or_offset_array=device_srcs,
            dst_or_index_array=device_dsts,
            weight_array=device_weights,
            store_transposed=False,
            renumber=True,
            vertices_array=device_vertices,
        )

    device_vertices = device_vertices.astype(dtype=np.int32)

    device_edge_ids = cp.asarray(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int64
    )

    with pytest.warns(UserWarning, match=warning_msg):
        SGGraph(
            resource_handle=resource_handle,
            graph_properties=graph_props,
            src_or_offset_array=device_srcs,
            dst_or_index_array=device_dsts,
            weight_array=device_weights,
            edge_id_array=device_edge_ids,
            store_transposed=False,
            renumber=True,
            vertices_array=device_vertices,
        )
