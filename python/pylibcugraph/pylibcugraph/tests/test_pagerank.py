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
# Pytest fixtures
# =============================================================================
# sg_graph_and_resource_handle fixture is defined in conftest.py. That fixture
# returns a preconstructed graph and corresponding resource handle for
# different datasets.
@pytest.fixture
def input_and_expected_output(sg_graph_objs):
    (g, resource_handle, ds_name) = sg_graph_objs
    return (g, resource_handle)


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.skip(reason="UNFINISHED")
def test_pagerank(input_and_expected_output):
    from pylibcugraph.experimental import pagerank

    (g, resource_handle) = input_and_expected_output

    precomputed_vertex_out_weight_sums = None
    alpha = 0.95
    epsilon = 0.0001
    max_iterations = 20
    has_initial_guess = False
    do_expensive_check = False

    result = pagerank(resource_handle,
                      g,
                      precomputed_vertex_out_weight_sums,
                      alpha,
                      epsilon,
                      max_iterations,
                      has_initial_guess,
                      do_expensive_check)
    print(result)
    raise NotImplementedError
