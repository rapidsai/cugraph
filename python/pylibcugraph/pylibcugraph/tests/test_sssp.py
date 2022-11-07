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


# =============================================================================
# Test data
# =============================================================================

# Map the names of input data to expected pagerank output
# The result names correspond to the datasets defined in conftest.py
_test_data = {
    "karate.csv": {
        "start_vertex": 1,
        "vertex": cp.asarray(range(34), dtype=np.int32),
        "distance": cp.asarray(
            [
                1.0,
                0.0,
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                1.0,
                3.0,
                3.0,
                3.0,
                1.0,
                3.0,
                1.0,
                3.0,
                1.0,
                3.0,
                3.0,
                3.0,
                3.0,
                3.0,
                2.0,
                2.0,
                3.0,
                1.0,
                2.0,
                2.0,
                2.0,
            ],
            dtype=np.float32,
        ),
        "predecessor": cp.asarray(
            [
                1,
                -1,
                1,
                1,
                0,
                0,
                0,
                1,
                0,
                2,
                0,
                0,
                0,
                1,
                32,
                32,
                5,
                1,
                32,
                1,
                32,
                1,
                32,
                32,
                27,
                31,
                33,
                2,
                2,
                32,
                1,
                0,
                2,
                13,
            ],
            dtype=np.int32,
        ),
    },
    "dolphins.csv": {
        "start_vertex": 1,
        "vertex": cp.asarray(range(62), dtype=np.int32),
        "distance": cp.asarray(
            [
                3.0,
                0.0,
                4.0,
                3.0,
                4.0,
                3.0,
                2.0,
                2.0,
                2.0,
                2.0,
                3.0,
                4.0,
                4.0,
                2.0,
                3.0,
                3.0,
                3.0,
                1.0,
                3.0,
                1.0,
                2.0,
                3.0,
                2.0,
                2.0,
                4.0,
                2.0,
                1.0,
                1.0,
                1.0,
                4.0,
                2.0,
                2.0,
                3.0,
                3.0,
                3.0,
                5.0,
                1.0,
                2.0,
                3.0,
                2.0,
                2.0,
                1.0,
                3.0,
                3.0,
                3.0,
                3.0,
                4.0,
                2.0,
                3.0,
                4.0,
                3.0,
                3.0,
                3.0,
                4.0,
                1.0,
                4.0,
                3.0,
                2.0,
                4.0,
                2.0,
                4.0,
                3.0,
            ],
            dtype=np.float32,
        ),
        "predecessor": cp.asarray(
            [
                40,
                -1,
                10,
                59,
                51,
                13,
                54,
                54,
                28,
                41,
                47,
                51,
                33,
                41,
                37,
                40,
                37,
                1,
                20,
                1,
                28,
                37,
                17,
                36,
                45,
                17,
                1,
                1,
                1,
                10,
                19,
                17,
                9,
                37,
                37,
                29,
                1,
                36,
                20,
                36,
                36,
                1,
                30,
                37,
                20,
                23,
                43,
                28,
                57,
                34,
                20,
                23,
                40,
                43,
                1,
                51,
                6,
                41,
                38,
                36,
                32,
                37,
            ],
            dtype=np.int32,
        ),
    },
    "Simple_1": {
        "start_vertex": 1,
        "vertex": cp.asarray(range(4), dtype=np.int32),
        "distance": cp.asarray(
            [
                3.4028235e38,
                0.0000000e00,
                1.0000000e00,
                2.0000000e00,
            ],
            dtype=np.float32,
        ),
        "predecessor": cp.asarray(
            [
                -1,
                -1,
                1,
                2,
            ],
            dtype=np.int32,
        ),
    },
    "Simple_2": {
        "start_vertex": 1,
        "vertex": cp.asarray(range(6), dtype=np.int32),
        "distance": cp.asarray(
            [
                3.4028235e38,
                0.0000000e00,
                3.4028235e38,
                2.0999999e00,
                1.1000000e00,
                4.3000002e00,
            ],
            dtype=np.float32,
        ),
        "predecessor": cp.asarray(
            [
                -1,
                -1,
                -1,
                1,
                1,
                4,
            ],
            dtype=np.int32,
        ),
    },
}

# =============================================================================
# Pytest fixtures
# =============================================================================
# fixtures used in this test module are defined in conftest.py


# =============================================================================
# Helper functions
# =============================================================================


# =============================================================================
# Tests
# =============================================================================
def test_sssp(sg_graph_objs):
    from pylibcugraph import sssp

    (g, resource_handle, ds_name) = sg_graph_objs

    (source, expected_verts, expected_distances, expected_predecessors) = _test_data[
        ds_name
    ].values()

    cutoff = 999999999  # maximum edge weight sum to consider
    compute_predecessors = True
    do_expensive_check = False

    result = sssp(
        resource_handle, g, source, cutoff, compute_predecessors, do_expensive_check
    )

    num_expected_verts = len(expected_verts)
    (actual_verts, actual_distances, actual_predecessors) = result

    # Do a simple check using the vertices as array indices.  First, ensure
    # the test data vertices start from 0 with no gaps.
    assert sum(range(num_expected_verts)) == sum(expected_verts)

    assert actual_verts.dtype == expected_verts.dtype
    assert actual_distances.dtype == expected_distances.dtype
    assert actual_predecessors.dtype == expected_predecessors.dtype

    actual_verts = actual_verts.tolist()
    actual_distances = actual_distances.tolist()
    actual_predecessors = actual_predecessors.tolist()
    expected_distances = expected_distances.tolist()
    expected_predecessors = expected_predecessors.tolist()

    for i in range(num_expected_verts):
        actual_distance = actual_distances[i]
        expected_distance = expected_distances[actual_verts[i]]
        # The distance value will be a MAX value (2**128) if there is no
        # predecessor, so only do a closer compare if either the actual or
        # expected are not that MAX value.
        if (actual_distance <= 3.4e38) or (expected_distance <= 3.4e38):
            assert actual_distance == pytest.approx(
                expected_distance, 1e-4
            ), f"actual != expected for distance result at index {i}"

        # The array of predecessors for graphs with multiple paths that are
        # equally short are non-deterministic, so skip those checks for
        # specific graph inputs.
        # FIXME: add a helper to verify paths are correct when results are
        # valid but non-deterministic
        if ds_name not in ["karate.csv", "dolphins.csv"]:
            assert actual_predecessors[i] == pytest.approx(
                expected_predecessors[actual_verts[i]], 1e-4
            ), f"actual != expected for predecessor result at index {i}"
