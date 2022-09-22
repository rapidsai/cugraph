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
from pylibcugraph import (SGGraph,
<<<<<<< HEAD
                          MGGraph,
=======
>>>>>>> upstream/branch-22.10
                          ResourceHandle,
                          GraphProperties,
                          )
from pylibcugraph import uniform_neighbor_sample


# =============================================================================
# Pytest fixtures
# =============================================================================
# fixtures used in this test module are defined in conftest.py


# =============================================================================
# Tests
# =============================================================================


def check_edges(result, srcs, dsts, weights, num_verts, num_edges, num_seeds):
    result_srcs, result_dsts, result_indices = result

    h_src_arr = srcs
    h_dst_arr = dsts
    h_wgt_arr = weights

    if isinstance(h_src_arr, cp.ndarray):
        h_src_arr = h_src_arr.get()
    if isinstance(h_dst_arr, cp.ndarray):
        h_dst_arr = h_dst_arr.get()
    if isinstance(h_wgt_arr, cp.ndarray):
        h_wgt_arr = h_wgt_arr.get()

    h_result_srcs = result_srcs.get()
    h_result_dsts = result_dsts.get()
    h_result_indices = result_indices.get()

    # Following the C validation, we will check that all edges are part of the
    # graph
    M = np.zeros((num_verts, num_verts), dtype=np.float32)

    # Construct the adjacency matrix
    for idx in range(num_edges):
        M[h_dst_arr[idx]][h_src_arr[idx]] = h_wgt_arr[idx]

    for edge in range(len(h_result_indices)):
        assert M[h_result_dsts[edge]][h_result_srcs[edge]] == \
            h_result_indices[edge]


# TODO: Coverage for the MG implementation
@pytest.mark.skipif(reason="skipping for testing purposes")
@pytest.mark.parametrize("renumber", [True, False])
@pytest.mark.parametrize("store_transposed", [True, False])
@pytest.mark.parametrize("with_replacement", [True, False])
def test_neighborhood_sampling_cupy(sg_graph_objs,
                                    valid_graph_data,
                                    renumber,
                                    store_transposed,
                                    with_replacement):

    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)

    device_srcs, device_dsts, device_weights, ds_name, is_valid = \
        valid_graph_data
    start_list = cp.random.choice(device_srcs, size=3)
    fanout_vals = np.asarray([1, 2], dtype="int32")

    # FIXME cupy has no attribute cp.union1d
    vertices = np.union1d(cp.asnumpy(device_srcs), cp.asnumpy(device_dsts))
    vertices = cp.asarray(vertices)
    num_verts = len(vertices)
    num_edges = max(len(device_srcs), len(device_dsts))

    sg = SGGraph(resource_handle,
                 graph_props,
                 device_srcs,
                 device_dsts,
                 device_weights,
                 store_transposed=store_transposed,
                 renumber=renumber,
                 do_expensive_check=False)

    result = uniform_neighbor_sample(resource_handle,
                                     sg,
                                     start_list,
                                     fanout_vals,
                                     with_replacement=with_replacement,
                                     do_expensive_check=False)

    check_edges(
        result, device_srcs, device_dsts, device_weights,
        num_verts, num_edges, len(start_list))


# TODO: Coverage for the MG implementation
@pytest.mark.skipif(reason="skipping for testing purposes")
@pytest.mark.parametrize("renumber", [True, False])
@pytest.mark.parametrize("store_transposed", [True, False])
@pytest.mark.parametrize("with_replacement", [True, False])
def test_neighborhood_sampling_cudf(sg_graph_objs,
                                    valid_graph_data,
                                    renumber,
                                    store_transposed,
                                    with_replacement):

    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)

    device_srcs, device_dsts, device_weights, ds_name, is_valid = \
        valid_graph_data
    # FIXME cupy has no attribute cp.union1d
    vertices = np.union1d(cp.asnumpy(device_srcs), cp.asnumpy(device_dsts))
    vertices = cp.asarray(vertices)

    device_srcs = cudf.Series(device_srcs, dtype=device_srcs.dtype)
    device_dsts = cudf.Series(device_dsts, dtype=device_dsts.dtype)
    device_weights = cudf.Series(device_weights, dtype=device_weights.dtype)

    start_list = cp.random.choice(device_srcs, size=3)
    fanout_vals = np.asarray([1, 2], dtype="int32")

    num_verts = len(vertices)
    num_edges = max(len(device_srcs), len(device_dsts))

    sg = SGGraph(resource_handle,
                 graph_props,
                 device_srcs,
                 device_dsts,
                 device_weights,
                 store_transposed=store_transposed,
                 renumber=renumber,
                 do_expensive_check=False)

    result = uniform_neighbor_sample(resource_handle,
                                     sg,
                                     start_list,
                                     fanout_vals,
                                     with_replacement=with_replacement,
                                     do_expensive_check=False)

    check_edges(
        result, device_srcs, device_dsts, device_weights,
        num_verts, num_edges, len(start_list))


def test_neighborhood_sampling_buffer():
    """
    """
    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)

    device_srcs = cp.asarray([0, 1, 1, 2, 2, 2, 3, 4], dtype=np.int32)
    device_dsts = cp.asarray([1, 3, 4, 0, 1, 3, 5, 5], dtype=np.int32)
    device_weights = cp.asarray([0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2],
                                dtype=np.float32)
    start_list = cp.asarray([2, 2], dtype=np.int32)
    fanout_vals = np.asarray([1, 2], dtype=np.int32)

    sg = SGGraph(resource_handle,
                 graph_props,
                 device_srcs,
                 device_dsts,
                 device_weights,
                 store_transposed=True,
                 renumber=False,
                 do_expensive_check=False)

    result = uniform_neighbor_sample(resource_handle,
                                     sg,
                                     start_list,
                                     fanout_vals,
                                     with_replacement=True,
                                     do_expensive_check=False)
    assert type(result) is tuple
