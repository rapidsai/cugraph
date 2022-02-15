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

# import pytest
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
                  "offsets": cp.asarray([], dtype=np.int32)
                  },
              "dolphins.csv": {
                  "seeds": cp.asarray([0, 0], dtype=np.int32),
                  "paths": cp.asarray([0, 14, 34, 49, 0, 42, 0, 40],
                                      dtype=np.int32),
                  "weights": cp.asarray([1., 1., 1., 1., 1., 1., 0., 0.],
                                        dtype=np.float32),
                  "offsets": cp.asarray([], dtype=np.int32)
                  },
              "Simple_1": {
                  "seeds": cp.asarray([0, 0], dtype=np.int32),
                  "paths": cp.asarray([0, 1, 2, 3, 0, 1, 2, 3],
                                      dtype=np.int32),
                  "weights": cp.asarray([1., 1., 1., 1., 1., 1., 0., 0.],
                                        dtype=np.float32),
                  "offsets": cp.asarray([], dtype=np.int32)
                  },
              "Simple_2": {
                  "seeds": cp.asarray([0, 0], dtype=np.int32),
                  "paths": cp.asarray([0, 1, 3, 5, 0, 1, 3, 5],
                                      dtype=np.int32),
                  "weights": cp.asarray([0.1, 2.1, 7.2, 0.1, 2.1, 7.2, 0., 0.],
                                        dtype=np.float32),
                  "offsets": cp.asarray([], dtype=np.int32)
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

    (seeds, expected_paths, expected_weights, expected_offsets) = \
        _test_data[ds_name].values()

    seeds = cp.asarray([0, 0], dtype=np.int32)

    max_depth = 4
    compress_result = False
    p = 0.8
    q = 0.5

    result = node2vec(resource_handle,
                      g,
                      seeds,
                      max_depth,
                      compress_result,
                      p,
                      q)

    (actual_paths, actual_weights, actual_offsets) = result

    # NOTE: This is not the actual check, but regardless should be expected to
    # fail at current moment
    num_walks = len(actual_paths)
    for i in range(num_walks):
        assert actual_paths[i] == expected_paths[i]
        assert actual_weights[i] == expected_weights[i]
