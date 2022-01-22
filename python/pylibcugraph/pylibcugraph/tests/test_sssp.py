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
# import cupy as cp
# import numpy as np


# =============================================================================
# Test data
# =============================================================================

# Map the names of input data to expected pagerank output
# The result names correspond to the datasets defined in conftest.py
_test_data = {"karate.csv":
              None,
              "dolphins.csv":
              None,
              "Simple_1":
              None,
              "Simple_2":
              None,
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
    expected_result = _test_data[ds_name]

    do_expensive_check = False

    result = sssp(resource_handle,
                  g,
                  do_expensive_check)

    assert result == expected_result
