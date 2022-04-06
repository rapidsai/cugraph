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
# from pylibcugraph.experimental import (ResourceHandle,
#                                        GraphProperties,
#                                        SGGraph,
#                                        node2vec)
# from cugraph.tests import utils
# import cugraph


# =============================================================================
# Test helpers
# =============================================================================
def _get_param_args(param_name, param_values):
    """
    Returns a tuple of (<param_name>, <pytest.param list>) which can be applied
    as the args to pytest.mark.parametrize(). The pytest.param list also
    contains param id string formed from teh param name and values.
    """
    return (param_name,
            [pytest.param(v, id=f"{param_name}={v}") for v in param_values])


def _generic_katz_test(src_arr,
                       dst_arr,
                       wgt_arr,
                       result_arr,
                       num_vertices,
                       num_edges,
                       store_transposed,
                       alpha,
                       beta,
                       epsilon,
                       max_iterations):
    """
    Builds a graph from the input arrays and runs katz using the other args,
    similar to how katz is tested in libcugraph.
    """
    # resource_handle = ResourceHandle()
    # graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)
    # G = SGGraph(resource_handle, graph_props, src_arr, dst_arr, wgt_arr,
    #             store_transposed=False, renumber=False,
    #             do_expensive_check=True)

    # (vertices, centralities) = katz_centrality(resource_handle, G, alpha,
    # beta, epsilon, max_iter, do_expensive_check)

    result_arr = result_arr.get()
    dud_result = [0, 0, 0, 0, 0, 0]

    for idx in range(num_vertices):
        expected_result = result_arr[idx]
        actual_result = dud_result[idx]
        if pytest.approx(expected_result, 1e-4) != actual_result:
            raise ValueError(f"Vertex {idx} has centrality {actual_result}"
                             f", should have been {expected_result}")


def test_katz():
    num_edges = 8
    num_vertices = 6
    src = cp.asarray([0, 1, 1, 2, 2, 2, 3, 4], dtype=np.int32)
    dst = cp.asarray([1, 3, 4, 0, 1, 3, 5, 5], dtype=np.int32)
    wgt = cp.asarray([0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2],
                     dtype=np.float32)
    result = cp.asarray([0.410614, 0.403211, 0.390689, 0.415175, 0.395125,
                        0.433226], dtype=np.float32)

    alpha = 0.01
    beta = 1.0
    epsilon = 0.000001
    max_iterations = 1000

    # Katz requires store_transposed to be True
    _generic_katz_test(src, dst, wgt, result, num_vertices, num_edges, True,
                       alpha, beta, epsilon, max_iterations)
