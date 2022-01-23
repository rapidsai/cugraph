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
_test_data = {"karate.csv": {
                  "start_vertex": 1,
                  "vertex": cp.asarray(range(34), dtype=np.int32),
                  "distance": cp.asarray(
                      [1., 0., 1., 1., 2., 2., 2., 1., 2., 2., 2., 2., 2., 1.,
                       3., 3., 3., 1., 3., 1., 3., 1., 3., 3., 3., 3., 3., 2.,
                       2., 3., 1., 2., 2., 2.,
                       ],
                      dtype=np.float32),
                  "predecessor": cp.asarray(
                      [1, -1, 1, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 1, 32,
                       32, 5, 1, 32, 1, 32, 1, 32, 32, 27, 31, 33, 2, 2, 32,
                       1, 0, 2, 13,
                       ],
                      dtype=np.int32),
                  },
              "dolphins.csv": {
                  "start_vertex": 1,
                  "vertex": cp.asarray(range(62), dtype=np.int32),
                  "distance": cp.asarray(
                      [3., 0., 4., 3., 4., 3., 2., 2., 2., 2., 3., 4., 4., 2.,
                       3., 3., 3., 1., 3., 1., 2., 3., 2., 2., 4., 2., 1., 1.,
                       1., 4., 2., 2., 3., 3., 3., 5., 1., 2., 3., 2., 2., 1.,
                       3., 3., 3., 3., 4., 2., 3., 4., 3., 3., 3., 4., 1., 4.,
                       3., 2., 4., 2., 4., 3.,
                       ],
                      dtype=np.float32),
                  "predecessor": cp.asarray(
                      [40, -1, 10, 59, 51, 13, 54, 54, 28, 41, 47, 51, 33, 41,
                       37, 40, 37, 1, 20, 1, 28, 37, 17, 36, 45, 17, 1, 1,
                       1, 10, 19, 17, 9, 37, 37, 29, 1, 36, 20, 36, 36, 1,
                       30, 37, 20, 23, 43, 28, 57, 34, 20, 23, 40, 43, 1, 51,
                       6, 41, 38, 36, 32, 37,
                       ],
                      dtype=np.int32),
                   },
              "Simple_1": {
                  "start_vertex": 1,
                  "vertex": cp.asarray(range(34), dtype=np.int32),
                  "distance": cp.asarray(
                      [3.4028235e+38, 0.0000000e+00, 1.0000000e+00,
                       2.0000000e+00,
                       ],
                      dtype=np.float32),
                  "predecessor": cp.asarray(
                      [-1, -1,  1,  2,
                       ],
                      dtype=np.int32),
                   },
              "Simple_2": {
                  "start_vertex": 1,
                  "vertex": cp.asarray(range(34), dtype=np.int32),
                  "distance": cp.asarray(
                      [3.4028235e+38, 0.0000000e+00, 3.4028235e+38,
                       2.0999999e+00, 1.1000000e+00, 4.3000002e+00
                       ],
                      dtype=np.float32),
                  "predecessor": cp.asarray(
                      [-1, -1, -1,  1,  1,  4,
                       ],
                      dtype=np.int32),
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
@pytest.mark.skip(reason="UNFINISHED")
def test_sssp(sg_graph_objs):
    from pylibcugraph.experimental import sssp

    (g, resource_handle, ds_name) = sg_graph_objs

    (source,
     expected_vertices,
     expected_distances,
     expected_predecessors) = _test_data[ds_name].values()

    cutoff = 999999999  # maximum edge weight sum to consider
    compute_predecessors = True
    do_expensive_check = False

    result = sssp(resource_handle,
                  g,
                  source,
                  cutoff,
                  compute_predecessors,
                  do_expensive_check)

    assert result == (expected_vertices,
                      expected_distances,
                      expected_predecessors)
