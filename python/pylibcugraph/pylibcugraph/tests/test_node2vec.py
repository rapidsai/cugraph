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

import pytest as pyt
import cupy as cp
import numpy as np


# =============================================================================
# Test data
# =============================================================================
# _alpha = 0.85
# _epsilon = 1.0e-6
# _max_iterations = 500

# The result names correspond to the datasets defined in conftest.py

_test_data = {"karate.csv": {
                  "seeds": cp.asarray([0, 0], dtype=np.int32),
                  "paths": cp.asarray([0, 8, 33, 29, 0, 1, 3, 0],
                                      dtype=np.int32),
                  "weights": cp.asarray([1., 1., 1., 1., 1., 1., 0., 0.],
                                        dtype=np.float32),
                  "offsets": cp.asarray([4, 4], dtype=np.int32),
                  "max_depth": 4
                  },
              "dolphins.csv": {
                  "seeds": cp.asarray([0, 0], dtype=np.int32),
                  "paths": cp.asarray([0, 14, 34, 49, 0, 42, 0, 40],
                                      dtype=np.int32),
                  "weights": cp.asarray([1., 1., 1., 1., 1., 1., 0., 0.],
                                        dtype=np.float32),
                  "offsets": cp.asarray([4, 4], dtype=np.int32),
                  "max_depth": 4
                  },
              "Simple_1": {
                  "seeds": cp.asarray([0, 3], dtype=np.int32),
                  "paths": cp.asarray([0, 1, 2, 3, 4, 4],
                                      dtype=np.int32),
                  "weights": cp.asarray([1., 1., 1., 1., 1., 1., 0., 0.],
                                        dtype=np.float32),
                  "offsets": cp.asarray([3, 3], dtype=np.int32),
                  "max_depth": 4
                  },
              "Simple_2": {
                  "seeds": cp.asarray([0, 3], dtype=np.int32),
                  "paths": cp.asarray([0, 1, 3, 5, 3, 5, 5, 5],
                                      dtype=np.int32),
                  "weights": cp.asarray([0.1, 2.1, 7.2, 0.1, 2.1, 7.2, 0., 0.],
                                        dtype=np.float32),
                  "offsets": cp.asarray([4, 4], dtype=np.int32),
                  "max_depth": 4
                  },
              }

# =============================================================================
# Pytest fixtures
# =============================================================================
# fixtures used in this test module are defined in conftest.py


# =============================================================================
# Tests
# =============================================================================
def test_node2vec(sg_graph_objs):
    from pylibcugraph.experimental import node2vec

    (g, resource_handle, ds_name) = sg_graph_objs

    # if ds_name not in ("Simple_1", "Simple_2"):
    #    return

    (seeds, expected_paths, expected_weights, expected_offsets, max_depth) = \
        _test_data[ds_name].values()

    compress_result = True
    p = 0.8
    q = 0.5

    result = node2vec(resource_handle, g, seeds, max_depth,
                      compress_result, p, q)

    (actual_paths, actual_weights, actual_offsets) = result
    num_walks = len(actual_paths)
    num_paths = len(seeds)

    # breakpoint()
    # Do a simple check using the vertices as array indices. First, ensure
    # the test data vertices start from 0 with no gaps.
    assert len(actual_offsets) == num_paths

    assert actual_paths.dtype == expected_paths.dtype
    assert actual_weights.dtype == expected_weights.dtype
    assert actual_offsets.dtype == expected_offsets.dtype

    actual_paths = actual_paths.tolist()
    actual_weights = actual_weights.tolist()
    actual_offsets = actual_offsets.tolist()
    expected_paths = expected_paths.tolist()
    expected_weights = expected_weights.tolist()
    expected_offsets = expected_offsets.tolist()

    if ds_name not in ["karate.csv", "dolphins.csv"]:
        for i in range(num_walks):
            assert pyt.approx(actual_paths[i], 1e-4) == expected_paths[i]
            assert pyt.approx(actual_weights[i], 1e-4) == expected_weights[i]

    # Starting vertex of each path should be the seed
    for i in range(num_paths):
        assert actual_paths[i*max_depth] == seeds[i]
