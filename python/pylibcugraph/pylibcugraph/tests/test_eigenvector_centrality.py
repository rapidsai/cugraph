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
from pylibcugraph import (
    ResourceHandle,
    GraphProperties,
    SGGraph,
    eigenvector_centrality,
)
from pylibcugraph.testing import utils


TOY = utils.RAPIDS_DATASET_ROOT_DIR_PATH / "toy_graph.csv"


# =============================================================================
# Test helpers
# =============================================================================
def _get_param_args(param_name, param_values):
    """
    Returns a tuple of (<param_name>, <pytest.param list>) which can be applied
    as the args to pytest.mark.parametrize(). The pytest.param list also
    contains param id string formed from the param name and values.
    """
    return (param_name, [pytest.param(v, id=f"{param_name}={v}") for v in param_values])


def _generic_eigenvector_test(
    src_arr,
    dst_arr,
    wgt_arr,
    result_arr,
    num_vertices,
    num_edges,
    store_transposed,
    epsilon,
    max_iterations,
):
    """
    Builds a graph from the input arrays and runs eigen using the other args,
    similar to how eigen is tested in libcugraph.
    """
    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)
    G = SGGraph(
        resource_handle,
        graph_props,
        src_arr,
        dst_arr,
        wgt_arr,
        store_transposed=False,
        renumber=False,
        do_expensive_check=True,
    )

    (vertices, centralities) = eigenvector_centrality(
        resource_handle, G, epsilon, max_iterations, do_expensive_check=False
    )

    result_arr = result_arr.get()
    vertices = vertices.get()
    centralities = centralities.get()

    for idx in range(num_vertices):
        vertex_id = vertices[idx]
        expected_result = result_arr[vertex_id]
        actual_result = centralities[idx]

        assert pytest.approx(expected_result, 1e-4) == actual_result, (
            f"Vertex {idx} has centrality {actual_result}, should have"
            f" been {expected_result}"
        )


def test_eigenvector():
    num_edges = 16
    num_vertices = 6
    graph_data = np.genfromtxt(TOY, delimiter=" ")
    src = cp.asarray(graph_data[:, 0], dtype=np.int32)
    dst = cp.asarray(graph_data[:, 1], dtype=np.int32)
    wgt = cp.asarray(graph_data[:, 2], dtype=np.float32)
    result = cp.asarray(
        [0.236325, 0.292055, 0.458457, 0.60533, 0.190498, 0.495942], dtype=np.float32
    )

    epsilon = 1e-6
    max_iterations = 200

    # Eigenvector requires store_transposed to be True?
    _generic_eigenvector_test(
        src, dst, wgt, result, num_vertices, num_edges, True, epsilon, max_iterations
    )
