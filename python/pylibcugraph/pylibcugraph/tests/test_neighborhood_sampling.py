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
_test_data = {"karate.csv": {"temp": cp.asarray([0], dtype=np.int32)},
              "dolphins.csv": {"temp": cp.asarray([0], dtype=np.int32)},
              "Simple_1": {"temp": cp.asarray([0], dtype=np.int32)},
              "Simple_2": {"temp": cp.asarray([0], dtype=np.int32)}
              }

# =============================================================================
# Pytest fixtures
# =============================================================================
# fixtures used in this test module are defined in conftest.py


# =============================================================================
# Tests
# =============================================================================

def test_neighborhood_sampling(sg_graph_objs):
    from pylibcugraph.experimental import uniform_neighborhood_sampling

    (g, resource_handle, ds_name) = sg_graph_objs

    (data) = _test_data[ds_name].values()
    # (path_tuples) = _test_data[ds_name].values()
    # (edge, rx_counts) = path_tuples 
    result = uniform_neighborhood_sampling(0,0,0,0)
    # result = uniform_neighborhood_sampling(G, start_info_list, fanout_vals, with_replacement=True)
    expected = data

    assert expected == data