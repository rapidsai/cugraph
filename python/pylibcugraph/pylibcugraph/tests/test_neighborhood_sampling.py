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

import gc

import pytest
import cupy as cp
import numpy as np
import cudf

from pylibcugraph import (SGGraph,
                          MGGraph,
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
    # FIXME: Update the result retrieval as the API changed
    result_srcs, result_dsts, result_indices = result
    h_src_arr = srcs.get()
    h_dst_arr = dsts.get()
    h_wgt_arr = weights.get()

    h_result_srcs = result_srcs.get()
    h_result_dsts = result_dsts.get()
    # FIXME: Variable not used
    # h_result_indices = result_indices.get()

    # Following the C validation, we will check that all edges are part of the
    # graph
    M = np.zeros((num_verts, num_verts), dtype=np.float64)

    for idx in range(num_edges):
        M[h_src_arr[idx]][h_dst_arr[idx]] = h_wgt_arr[idx]

    for edge in range(h_result_srcs):
        assert M[h_result_srcs[edge]][h_result_dsts[edge]] > 0.0
        # found = False
        for j in range(num_seeds):
            # FIXME: Revise, this is not correct.
            # Labels are no longer supported.
            # found = found or (h_result_labels[edge] == h_result_indices[j])
            pass


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
    fanout_vals = cp.asarray([1, 2], dtype=np.int32)

    mg = MGGraph(resource_handle,
                 graph_props,
                 device_srcs,
                 device_dsts,
                 device_weights,
                 store_transposed=True,
                 num_edges=8,
                 do_expensive_check=False)

    result = uniform_neighbor_sample(resource_handle,
                                     mg,
                                     start_list,
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
    fanout_vals = cudf.Series([1, 2], dtype=np.int32)

    mg = MGGraph(resource_handle,
                 graph_props,
                 device_srcs,
                 device_dsts,
                 device_weights,
                 store_transposed=True,
                 num_edges=8,
                 do_expensive_check=False)

    result = uniform_neighbor_sample(resource_handle,
                                     mg,
                                     start_list,
                                     fanout_vals,
                                     with_replacement=True,
                                     do_expensive_check=False)

    check_edges(result, device_srcs, device_dsts, device_weights, 6, 8, 2)


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
    device_dsts = cp.arange(1, 1e6+1, dtype=np.int32)
    device_weights = cp.asarray([1.0]*int(1e6), dtype=np.float32)

    # start_list == every vertex is intentionally excessive
    start_list = device_srcs
    fanout_vals = np.asarray([1, 2], dtype=np.int32)

    sg = SGGraph(resource_handle,
                 graph_props,
                 device_srcs,
                 device_dsts,
                 device_weights,
                 store_transposed=True,
                 renumber=False,
                 do_expensive_check=False)

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
        do_expensive_check=False)

    assert type(result) is tuple
    assert isinstance(result[0], cp.ndarray)
    assert isinstance(result[1], cp.ndarray)
    assert isinstance(result[2], cp.ndarray)
    # Crude check that the results are accessible
    assert result[0][0].dtype == np.int32
    assert result[1][0].dtype == np.int32
    assert result[2][0].dtype == np.float32

    # Cleanup the result - this should leave the memory used equal to the amount
    # prior to running the algo.
    free_before_cleanup = device.mem_info[0]
    print(f"{free_before_cleanup=}")
    result_size = (len(result[0]) + len(result[1]) + len(result[2])) * (32//8)
    del result
    gc.collect()
    free_after_cleanup = device.mem_info[0]
    print(f"{free_after_cleanup=}")
    actual_delta = free_after_cleanup - free_before_cleanup
    expected_delta = free_memory_before - free_before_cleanup
    leak = expected_delta - actual_delta
    print(f"  {result_size=} {actual_delta=} {expected_delta=} {leak=}")
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
    # intended for testing only - SampleResult objects are normally only created
    # by running a sampling algo.
    sampling_result = create_sampling_result(
        resource_handle,
        host_sources=np.arange(1e8, dtype="int32"),
        host_destinations=np.arange(1, 1e8+1, dtype="int32"),
        host_indices=np.arange(1e8, dtype="int32"),
    )

    assert free_memory_before > device.mem_info[0]

    sources = sampling_result.get_sources()
    destinations = sampling_result.get_destinations()
    indices = sampling_result.get_indices()

    assert isinstance(sources, cp.ndarray)
    assert isinstance(destinations, cp.ndarray)
    assert isinstance(indices, cp.ndarray)

    # Delete the SampleResult instance. This *should not* free the device memory
    # yet since the variables sources, destinations, and indices are keeping the
    # refcount >0.
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

    # delete the variables which should take the ref count on sampling_result to
    # 0, which will cause it to be garbage collected.
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
