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
_alpha = 0.85
_epsilon = 1.0e-6
_max_iterations = 500

# Map the names of input data to expected pagerank output
# The result names correspond to the datasets defined in conftest.py
_test_data = {
    "karate.csv": (
        cp.asarray(range(34), dtype=np.int32),
        cp.asarray(
            [
                0.096998,
                0.052877,
                0.057078,
                0.035860,
                0.021978,
                0.029111,
                0.029111,
                0.024491,
                0.029766,
                0.014309,
                0.021978,
                0.009565,
                0.014645,
                0.029536,
                0.014536,
                0.014536,
                0.016784,
                0.014559,
                0.014536,
                0.019605,
                0.014536,
                0.014559,
                0.014536,
                0.031522,
                0.021076,
                0.021006,
                0.015044,
                0.025640,
                0.019573,
                0.026288,
                0.024590,
                0.037158,
                0.071693,
                0.100919,
            ],
            dtype=np.float32,
        ),
    ),
    "dolphins.csv": (
        cp.asarray(range(62), dtype=np.int32),
        cp.asarray(
            [
                0.01696534,
                0.02465084,
                0.01333804,
                0.00962903,
                0.00507979,
                0.01442816,
                0.02005379,
                0.01564308,
                0.01709825,
                0.02345867,
                0.01510835,
                0.00507979,
                0.0048353,
                0.02615709,
                0.03214436,
                0.01988301,
                0.01662675,
                0.03172837,
                0.01939547,
                0.01292825,
                0.02464085,
                0.01693892,
                0.00541593,
                0.00986347,
                0.01690569,
                0.01150429,
                0.0112102,
                0.01713019,
                0.01484573,
                0.02645844,
                0.0153021,
                0.00541593,
                0.01330877,
                0.02842296,
                0.01591988,
                0.00491821,
                0.02061337,
                0.02987523,
                0.02393915,
                0.00776477,
                0.02196631,
                0.01613769,
                0.01761861,
                0.02169104,
                0.01283079,
                0.02951408,
                0.00882587,
                0.01733948,
                0.00526172,
                0.00887672,
                0.01923187,
                0.03129924,
                0.01207255,
                0.00818102,
                0.02165103,
                0.00749415,
                0.0083263,
                0.0300956,
                0.00496289,
                0.01476788,
                0.00619018,
                0.01103916,
            ],
            dtype=np.float32,
        ),
    ),
    "Simple_1": (
        cp.asarray(range(4), dtype=np.int32),
        cp.asarray([0.11615585, 0.21488841, 0.2988108, 0.3701449], dtype=np.float32),
    ),
    "Simple_2": (
        cp.asarray(range(6), dtype=np.int32),
        cp.asarray(
            [
                0.09902544,
                0.17307726,
                0.0732199,
                0.1905103,
                0.12379099,
                0.34037617,
            ],
            dtype=np.float32,
        ),
    ),
}

# =============================================================================
# Pytest fixtures
# =============================================================================
# fixtures used in this test module are defined in conftest.py


# =============================================================================
# Tests
# =============================================================================

# FIXME: add tests for non-transposed graphs too, which should either work (via
# auto-transposing in C) or raise the appropriate exception.


def test_pagerank(sg_transposed_graph_objs):
    from pylibcugraph import pagerank

    (g, resource_handle, ds_name) = sg_transposed_graph_objs
    (expected_verts, expected_pageranks) = _test_data[ds_name]

    precomputed_vertex_out_weight_sums = None
    do_expensive_check = False
    precomputed_vertex_out_weight_vertices = None
    precomputed_vertex_out_weight_sums = None
    initial_guess_vertices = None
    initial_guess_values = None

    result = pagerank(
        resource_handle,
        g,
        precomputed_vertex_out_weight_vertices,
        precomputed_vertex_out_weight_sums,
        initial_guess_vertices,
        initial_guess_values,
        _alpha,
        _epsilon,
        _max_iterations,
        do_expensive_check,
    )

    num_expected_verts = len(expected_verts)
    (actual_verts, actual_pageranks) = result

    # Do a simple check using the vertices as array indices.  First, ensure
    # the test data vertices start from 0 with no gaps.
    assert sum(range(num_expected_verts)) == sum(expected_verts)

    assert actual_verts.dtype == expected_verts.dtype
    assert actual_pageranks.dtype == expected_pageranks.dtype

    actual_pageranks = actual_pageranks.tolist()
    actual_verts = actual_verts.tolist()
    expected_pageranks = expected_pageranks.tolist()

    for i in range(num_expected_verts):
        assert actual_pageranks[i] == pytest.approx(
            expected_pageranks[actual_verts[i]], 1e-4
        ), f"actual != expected for result at index {i}"
