# Copyright (c) 2023, NVIDIA CORPORATION.
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
    edge_betweenness_centrality,
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


def _generic_edge_betweenness_centrality_test(
    src_arr,
    dst_arr,
    edge_id_arr,
    result_score_arr,
    result_edge_id_arr,
    num_edges,
    store_transposed,
    k,
    random_state,
    normalized,
):
    """
    Builds a graph from the input arrays and runs edge bc using the other args,
    similar to how edge bc is tested in libcugraph.
    """
    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)
    G = SGGraph(
        resource_handle,
        graph_props,
        src_arr,
        dst_arr,
        store_transposed=store_transposed,
        renumber=False,
        do_expensive_check=True,
        edge_id_array=edge_id_arr,
    )

    (_, _, values, edge_ids) = edge_betweenness_centrality(
        resource_handle, G, k, random_state, normalized, do_expensive_check=False
    )

    result_score_arr = result_score_arr.get()
    result_edge_id_arr = result_edge_id_arr.get()
    centralities = values.get()
    edge_ids = edge_ids.get()

    for idx in range(num_edges):
        expected_result_score = result_score_arr[idx]
        actual_result_score = centralities[idx]

        expected_result_edge_id = result_edge_id_arr[idx]
        actual_result_edge_id = edge_ids[idx]

        assert pytest.approx(expected_result_score, 1e-4) == actual_result_score, (
            f"Edge {src_arr[idx]} {dst_arr[idx]} has centrality {actual_result_score},"
            f" should have been {expected_result_score}"
        )

        assert pytest.approx(expected_result_edge_id, 1e-4) == actual_result_edge_id, (
            f"Edge {src_arr[idx]} {dst_arr[idx]} has id {actual_result_edge_id},"
            f" should have been {expected_result_edge_id}"
        )


def test_edge_betweenness_centrality():
    num_edges = 16

    graph_data = np.genfromtxt(TOY, delimiter=" ")
    src = cp.asarray(graph_data[:, 0], dtype=np.int32)
    dst = cp.asarray(graph_data[:, 1], dtype=np.int32)
    edge_id = cp.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int32
    )
    result_score = cp.asarray(
        [
            0.10555556,
            0.06111111,
            0.10555556,
            0.06666667,
            0.09444445,
            0.14444445,
            0.06111111,
            0.06666667,
            0.09444445,
            0.09444445,
            0.09444445,
            0.12222222,
            0.14444445,
            0.07777778,
            0.12222222,
            0.07777778,
        ],
        dtype=np.float32,
    )
    result_edge_ids = cp.asarray([0, 11, 8, 12, 1, 2, 3, 4, 5, 9, 13, 6, 10, 7, 14, 15])

    store_transposed = False
    k = None
    random_state = None
    normalized = True

    _generic_edge_betweenness_centrality_test(
        src,
        dst,
        edge_id,
        result_score,
        result_edge_ids,
        num_edges,
        store_transposed,
        k,
        random_state,
        normalized,
    )
