# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import gc

import pytest
import cupy as cp
import numpy as np
import cudf

from pylibcugraph import (
    SGGraph,
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
        assert M[h_result_dsts[edge]][h_result_srcs[edge]] == h_result_indices[edge]


# TODO: Coverage for the MG implementation
@pytest.mark.skipif(reason="skipping for testing purposes")
@pytest.mark.parametrize("renumber", [True, False])
@pytest.mark.parametrize("store_transposed", [True, False])
@pytest.mark.parametrize("with_replacement", [True, False])
def test_neighborhood_sampling_cupy(
    sg_graph_objs, valid_graph_data, renumber, store_transposed, with_replacement
):

    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)

    device_srcs, device_dsts, device_weights, ds_name, is_valid = valid_graph_data
    start_list = cp.random.choice(device_srcs, size=3)
    fanout_vals = np.asarray([1, 2], dtype="int32")

    # FIXME cupy has no attribute cp.union1d
    vertices = np.union1d(cp.asnumpy(device_srcs), cp.asnumpy(device_dsts))
    vertices = cp.asarray(vertices)
    num_verts = len(vertices)
    num_edges = max(len(device_srcs), len(device_dsts))

    sg = SGGraph(
        resource_handle,
        graph_props,
        device_srcs,
        device_dsts,
        device_weights,
        store_transposed=store_transposed,
        renumber=renumber,
        do_expensive_check=False,
    )

    result = uniform_neighbor_sample(
        resource_handle,
        sg,
        start_list,
        fanout_vals,
        with_replacement=with_replacement,
        do_expensive_check=False,
    )

    check_edges(
        result,
        device_srcs,
        device_dsts,
        device_weights,
        num_verts,
        num_edges,
        len(start_list),
    )


# TODO: Coverage for the MG implementation
@pytest.mark.skipif(reason="skipping for testing purposes")
@pytest.mark.parametrize("renumber", [True, False])
@pytest.mark.parametrize("store_transposed", [True, False])
@pytest.mark.parametrize("with_replacement", [True, False])
def test_neighborhood_sampling_cudf(
    sg_graph_objs, valid_graph_data, renumber, store_transposed, with_replacement
):

    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)

    device_srcs, device_dsts, device_weights, ds_name, is_valid = valid_graph_data
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

    sg = SGGraph(
        resource_handle,
        graph_props,
        device_srcs,
        device_dsts,
        device_weights,
        store_transposed=store_transposed,
        renumber=renumber,
        do_expensive_check=False,
    )

    result = uniform_neighbor_sample(
        resource_handle,
        sg,
        start_list,
        fanout_vals,
        with_replacement=with_replacement,
        do_expensive_check=False,
    )

    check_edges(
        result,
        device_srcs,
        device_dsts,
        device_weights,
        num_verts,
        num_edges,
        len(start_list),
    )


@pytest.mark.cugraph_ops
def test_neighborhood_sampling_large_sg_graph(gpubenchmark):
    """
    Use a large SG graph and set input args accordingly to test/benchmark
    returning a large result.
    """
    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)

    # FIXME: this graph is just a line - consider a better graph that exercises
    # neighborhood sampling better/differently
    device_srcs = cp.arange(1e6, dtype=np.int32)
    device_dsts = cp.arange(1, 1e6 + 1, dtype=np.int32)
    device_weights = cp.asarray([1.0] * int(1e6), dtype=np.float32)

    # start_list == every vertex is intentionally excessive
    start_list = device_srcs
    fanout_vals = np.asarray([1, 2], dtype=np.int32)

    sg = SGGraph(
        resource_handle,
        graph_props,
        device_srcs,
        device_dsts,
        device_weights,
        store_transposed=True,
        renumber=False,
        do_expensive_check=False,
    )

    # Ensure the only memory used after the algo call is for the result, so
    # take a snapshot here.
    # Assume GPU 0 will be used and the test has
    # exclusive access and nothing else can use its memory while the test is
    # running.
    gc.collect()
    device = cp.cuda.Device(0)
    free_memory_before = device.mem_info[0]

    result = gpubenchmark(
        uniform_neighbor_sample,
        resource_handle,
        sg,
        start_list,
        fanout_vals,
        with_replacement=True,
        do_expensive_check=False,
    )

    assert type(result) is tuple
    assert isinstance(result[0], cp.ndarray)
    assert isinstance(result[1], cp.ndarray)
    assert isinstance(result[2], cp.ndarray)
    # Crude check that the results are accessible
    assert result[0][0].dtype == np.int32
    assert result[1][0].dtype == np.int32
    assert result[2][0].dtype == np.float32

    # FIXME: this is to help debug a leak in uniform_neighbor_sample, remove
    # once leak is fixed
    free_before_cleanup = device.mem_info[0]
    print(f"{free_before_cleanup=}")

    result_bytes = (len(result[0]) + len(result[1]) + len(result[2])) * (32 // 8)

    # Cleanup the result - this should leave the memory used equal to the
    # amount prior to running the algo.
    del result
    gc.collect()

    # FIXME: this is to help debug a leak in uniform_neighbor_sample, remove
    # once leak is fixed
    free_after_cleanup = device.mem_info[0]
    print(f"{free_after_cleanup=}")
    actual_delta = free_after_cleanup - free_before_cleanup
    expected_delta = free_memory_before - free_before_cleanup
    leak = expected_delta - actual_delta
    print(f"  {result_bytes=} {actual_delta=} {expected_delta=} {leak=}")
    assert free_memory_before == device.mem_info[0]


def test_sample_result():
    """
    Ensure the SampleResult class returns zero-opy cupy arrays and properly
    frees device memory when all references to it are gone and it's garbage
    collected.
    """
    from pylibcugraph.testing.type_utils import create_sampling_result

    gc.collect()

    resource_handle = ResourceHandle()
    # Assume GPU 0 will be used and the test has exclusive access and nothing
    # else can use its memory while the test is running.
    device = cp.cuda.Device(0)
    free_memory_before = device.mem_info[0]

    # Use the testing utility to create a large sampling result.  This API is
    # intended for testing only - SampleResult objects are normally only
    # created by running a sampling algo.
    sampling_result = create_sampling_result(
        resource_handle,
        device_sources=cp.arange(1e8, dtype="int32"),
        device_destinations=cp.arange(1, 1e8 + 1, dtype="int32"),
        device_weights=cp.arange(1e8 + 2, dtype="float32"),
        device_edge_id=cp.arange(1e8 + 3, dtype="int32"),
        device_edge_type=cp.arange(1e8 + 4, dtype="int32"),
        device_hop=cp.arange(1e8 + 5, dtype="int32"),
        device_batch_label=cp.arange(1e8 + 6, dtype="int32"),
    )

    assert free_memory_before > device.mem_info[0]

    sources = sampling_result.get_sources()
    destinations = sampling_result.get_destinations()
    indices = sampling_result.get_indices()

    assert isinstance(sources, cp.ndarray)
    assert isinstance(destinations, cp.ndarray)
    assert isinstance(indices, cp.ndarray)

    # Delete the SampleResult instance. This *should not* free the device
    # memory yet since the variables sources, destinations, and indices are
    # keeping the refcount >0.
    del sampling_result
    gc.collect()
    assert free_memory_before > device.mem_info[0]

    # Check that the data is still valid
    assert sources[999] == 999
    assert destinations[999] == 1000
    assert indices[999] == 999

    # Add yet another reference to the original data, which should prevent it
    # from being freed when the GC runs.
    sources2 = sources

    # delete the variables which should take the ref count on sampling_result
    # to 0, which will cause it to be garbage collected.
    del sources
    del destinations
    del indices
    gc.collect()

    # sources2 should be keeping the data alive
    assert sources2[999] == 999
    assert free_memory_before > device.mem_info[0]

    # All memory should be freed once the last reference is deleted
    del sources2
    gc.collect()
    assert free_memory_before == device.mem_info[0]
