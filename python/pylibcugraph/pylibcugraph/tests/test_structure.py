# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import cupy as cp
import numpy as np
from pylibcugraph import (
    SGGraph,
    ResourceHandle,
    GraphProperties,
    decompress_to_edgelist,
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


@pytest.mark.parametrize(
    "input_array_format,vertex_dtype",
    [("COO", np.int32), ("COO", np.int64), ("CSR", np.int32)],
)
@pytest.mark.parametrize("renumber", [False, True])
@pytest.mark.parametrize("store_transposed", [False, True])
def test_decompress_preserves_edge_direction(
    input_array_format, renumber, store_transposed, vertex_dtype
):
    handle = ResourceHandle()
    sources = cp.asarray([0, 0, 1, 1, 2, 3], dtype=vertex_dtype)
    destinations = cp.asarray([1, 2, 2, 3, 3, 0], dtype=vertex_dtype)
    weights = cp.asarray([1, 4, 2, 7, 1, 8], dtype=np.float64)
    offsets = cp.asarray([0, 2, 4, 5, 6], dtype=np.int32)
    graph = SGGraph(
        resource_handle=handle,
        graph_properties=GraphProperties(is_symmetric=False, is_multigraph=False),
        src_or_offset_array=offsets if input_array_format == "CSR" else sources,
        dst_or_index_array=destinations,
        weight_array=weights,
        input_array_format=input_array_format,
        store_transposed=store_transposed,
        renumber=renumber,
        do_expensive_check=True,
    )
    # Repeated extraction must preserve the logical edges for either storage layout.
    for _ in range(2):
        src, dst, weight, _, _ = decompress_to_edgelist(handle, graph, True)
        order = cp.lexsort(cp.stack((dst, src)))
        cp.testing.assert_array_equal(src[order], sources)
        cp.testing.assert_array_equal(dst[order], destinations)
        cp.testing.assert_array_equal(weight[order], weights)
        assert src.dtype == sources.dtype
        assert weight.dtype == weights.dtype
