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
        "The graph requires the 'vertices_array' values "
        "to match the 'src_or_offset_array' and 'dst_or_index_array'. "
        f"'vertices_array' type is: {device_vertices.dtype} "
        f"'src_or_offset_array' type is: {device_srcs.dtype} and "
        f"'dst_or_index_array' type is : {device_dsts.dtype}."
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

    warning_msg = (
        "The graph requires the 'edge_id_array' values "
        "to match the 'src_or_offset_array' and 'dst_or_index_array'. "
        f"'edge_id_array' type is: {device_edge_ids.dtype} "
        f"'src_or_offset_array' type is: {device_srcs.dtype} and "
        f"'dst_or_index_array' type is : {device_dsts.dtype}."
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

    device_edge_ids = device_edge_ids.astype(dtype=np.int32)

    device_edge_start_times = cp.asarray(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=np.int32
    )

    device_edge_end_times = cp.asarray(
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype=np.int64
    )

    warning_msg = (
        "The graph requires the 'edge_start_time_array' values "
        "to match the 'edge_end_time_array' type. "
        f"'edge_start_time_array' type is: {device_edge_start_times.dtype} and "
        f"'edge_end_time_array' type is : {device_edge_end_times.dtype}."
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
            edge_start_time_array=device_edge_start_times,
            edge_end_time_array=device_edge_end_times,
            vertices_array=device_vertices,
        )
