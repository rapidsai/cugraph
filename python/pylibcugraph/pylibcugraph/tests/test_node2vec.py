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
# The result names correspond to the datasets defined in conftest.py

_test_data = {"karate.csv": {
                  "seeds": cp.asarray([0, 0], dtype=np.int32),
                  "paths": cp.asarray([0, 8, 33, 29, 26, 0, 1, 3, 13, 33],
                                      dtype=np.int32),
                  "weights": cp.asarray([1., 1., 1., 1., 1., 1., 1., 1.],
                                        dtype=np.float32),
                  "path_sizes": cp.asarray([5, 5], dtype=np.int32),
                  "max_depth": 5
                  },
              "dolphins.csv": {
                  "seeds": cp.asarray([11], dtype=np.int32),
                  "paths": cp.asarray([11, 51, 11, 51],
                                      dtype=np.int32),
                  "weights": cp.asarray([1., 1., 1.],
                                        dtype=np.float32),
                  "path_sizes": cp.asarray([4], dtype=np.int32),
                  "max_depth": 4
                  },
              "Simple_1": {
                  "seeds": cp.asarray([0, 3], dtype=np.int32),
                  "paths": cp.asarray([0, 1, 2, 3],
                                      dtype=np.int32),
                  "weights": cp.asarray([1., 1.],
                                        dtype=np.float32),
                  "path_sizes": cp.asarray([3, 1], dtype=np.int32),
                  "max_depth": 3
                  },
              "Simple_2": {
                  "seeds": cp.asarray([0, 3], dtype=np.int32),
                  "paths": cp.asarray([0, 1, 3, 5, 3, 5],
                                      dtype=np.int32),
                  "weights": cp.asarray([0.1, 2.1, 7.2, 7.2],
                                        dtype=np.float32),
                  "path_sizes": cp.asarray([4, 2], dtype=np.int32),
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
# def test_node2vec_untransposed(sg_graph_objs):
#    return test_node2vec(sg_graph_objs)

# TODO: Create test data for transposed graphs
# def test_node2vec_transposed(sg_transposed_graph_objs):
#    return test_node2vec(sg_transposed_graph_objs)

@pytest.mark.parametrize("compress_result", [True, False])
def test_node2vec(sg_graph_objs, compress_result):
    from pylibcugraph.experimental import node2vec

    (g, resource_handle, ds_name) = sg_graph_objs

    (seeds, expected_paths, expected_weights, expected_path_sizes, max_depth) \
        = _test_data[ds_name].values()

    p = 0.8
    q = 0.5

    result = node2vec(resource_handle, g, seeds, max_depth,
                      compress_result, p, q)

    (actual_paths, actual_weights, actual_path_sizes) = result
    num_paths = len(seeds)

    # Verify that the correct number of paths were made
    if compress_result:
        assert len(actual_path_sizes) == num_paths
        assert actual_path_sizes.dtype == expected_path_sizes.dtype
        actual_path_sizes = actual_path_sizes.tolist()
        expected_path_sizes = expected_path_sizes.tolist()
        expected_walks = sum(expected_path_sizes) - num_paths
        # FIXME: When using multiple seeds, paths are connected via the weights
        # array, there should not be a weight connecting the end of a path with
        # the beginning of another. PR #2089 will resolve this.
        # Verify the number of walks was equal to path sizes - num paths
        assert len(actual_weights) == expected_walks

    assert actual_paths.dtype == expected_paths.dtype
    assert actual_weights.dtype == expected_weights.dtype
    actual_paths = actual_paths.tolist()
    actual_weights = actual_weights.tolist()
    expected_paths = expected_paths.tolist()
    expected_weights = expected_weights.tolist()

    # Verify exact walks chosen for linear graph Simple_1
    if ds_name not in ["karate.csv", "dolphins.csv", "Simple_2"]:
        for i in range(len(expected_paths)):
            assert pytest.approx(actual_paths[i], 1e-4) == expected_paths[i]
        for i in range(len(expected_weights)):
            assert pytest.approx(actual_weights[i], 1e-4) == expected_weights[i]

    # Verify starting vertex of each path is the corresponding seed
    if compress_result:
        path_start = 0
        for i in range(num_paths):
            assert actual_path_sizes[i] == expected_path_sizes[i]
            assert actual_paths[path_start] == seeds[i]
            path_start += actual_path_sizes[i]
