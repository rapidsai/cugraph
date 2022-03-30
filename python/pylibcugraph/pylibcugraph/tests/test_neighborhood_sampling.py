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
import cudf
from pylibcugraph.experimental import (MGGraph,
                                       ResourceHandle,
                                       GraphProperties,
                                       uniform_neighborhood_sampling,
                                       )


# =============================================================================
# Pytest fixtures
# =============================================================================
# fixtures used in this test module are defined in conftest.py


# =============================================================================
# Tests
# =============================================================================


def check_edges(result, srcs, dsts, weights, num_verts, num_edges, num_seeds):
    result_srcs, result_dsts, result_labels, result_indices = result
    h_src_arr = srcs.get()
    h_dst_arr = dsts.get()
    h_wgt_arr = weights.get()

    h_result_srcs = result_srcs.get()
    h_result_dsts = result_dsts.get()
    h_result_labels = result_labels.get()
    h_result_indices = result_indices.get()

    # Following the C validation, we will check that all edges are part of the
    # graph
    M = np.zeros((num_verts, num_verts), dtype=np.float64)

    for idx in range(num_edges):
        M[h_src_arr[idx]][h_dst_arr[idx]] = h_wgt_arr[idx]

    for edge in range(h_result_srcs):
        assert M[h_result_srcs[edge]][h_result_dsts[edge]] > 0.0
        found = False
        for j in range(num_seeds):
            # Revise, this is not correct
            found = found or (h_result_labels[edge] == h_result_indices[j])


# TODO: Refactor after creating a helper within conftest.py to pass in an
# mg_graph_objs instance
@pytest.mark.skip(reason="pylibcugraph MG test infra not complete")
def test_neighborhood_sampling_cupy():
    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)

    device_srcs = cp.asarray([0, 1, 1, 2, 2, 2, 3, 4], dtype=np.int32)
    device_dsts = cp.asarray([1, 3, 4, 0, 1, 3, 5, 5], dtype=np.int32)
    device_weights = cp.asarray([0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2],
                                dtype=np.float32)
    start_list = cp.asarray([2, 2], dtype=np.int32)
    info_list = cp.asarray([0, 1], dtype=np.int32)
    fanout_vals = cp.asarray([1, 2], dtype=np.int32)

    mg = MGGraph(resource_handle,
                 graph_props,
                 device_srcs,
                 device_dsts,
                 device_weights,
                 store_transposed=True,
                 num_edges=8,
                 do_expensive_check=False)

    result = uniform_neighborhood_sampling(resource_handle,
                                           mg,
                                           start_list,
                                           info_list,
                                           fanout_vals,
                                           with_replacement=True,
                                           do_expensive_check=False)

    check_edges(result, device_srcs, device_dsts, device_weights, 6, 8, 2)


@pytest.mark.skip(reason="pylibcugraph MG test infra not complete")
def test_neighborhood_sampling_cudf():
    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)

    device_srcs = cudf.Series([0, 1, 1, 2, 2, 2, 3, 4], dtype=np.int32)
    device_dsts = cudf.Series([1, 3, 4, 0, 1, 3, 5, 5], dtype=np.int32)
    device_weights = cudf.Series([0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2],
                                 dtype=np.float32)
    start_list = cudf.Series([2, 2], dtype=np.int32)
    info_list = cudf.Series([0, 1], dtype=np.int32)
    fanout_vals = cudf.Series([1, 2], dtype=np.int32)

    mg = MGGraph(resource_handle,
                 graph_props,
                 device_srcs,
                 device_dsts,
                 device_weights,
                 store_transposed=True,
                 num_edges=8,
                 do_expensive_check=False)

    result = uniform_neighborhood_sampling(resource_handle,
                                           mg,
                                           start_list,
                                           info_list,
                                           fanout_vals,
                                           with_replacement=True,
                                           do_expensive_check=False)

    check_edges(result, device_srcs, device_dsts, device_weights, 6, 8, 2)
