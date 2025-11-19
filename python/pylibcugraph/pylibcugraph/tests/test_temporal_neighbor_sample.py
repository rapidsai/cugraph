# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import cupy as cp
import pytest

from pylibcugraph import (
    ResourceHandle,
    GraphProperties,
    SGGraph,
    homogeneous_uniform_temporal_neighbor_sample,
    homogeneous_biased_temporal_neighbor_sample,
    heterogeneous_uniform_temporal_neighbor_sample,
    heterogeneous_biased_temporal_neighbor_sample,
)


def _build_temporal_sg(resource_handle: ResourceHandle) -> SGGraph:
    srcs = cp.asarray([0, 1, 1, 2], dtype=np.int32)
    dsts = cp.asarray([1, 2, 3, 3], dtype=np.int32)
    weights = cp.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    edge_start_times = cp.asarray([0, 1, 2, 3], dtype=np.int32)
    edge_end_times = cp.asarray([1, 2, 3, 4], dtype=np.int32)

    props = GraphProperties(is_symmetric=False, is_multigraph=False)
    G = SGGraph(
        resource_handle,
        props,
        srcs,
        dsts,
        weight_array=weights,
        edge_start_time_array=edge_start_times,
        edge_end_time_array=edge_end_times,
        store_transposed=True,
        renumber=False,
        do_expensive_check=False,
    )
    return G


def _build_temporal_sg_with_edge_types(resource_handle: ResourceHandle) -> SGGraph:
    srcs = cp.asarray([0, 1, 1, 2], dtype=np.int32)
    dsts = cp.asarray([1, 2, 3, 3], dtype=np.int32)
    weights = cp.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    edge_start_times = cp.asarray([0, 1, 2, 3], dtype=np.int32)
    edge_end_times = cp.asarray([1, 2, 3, 4], dtype=np.int32)
    edge_types = cp.asarray([0, 1, 0, 1], dtype=np.int32)

    props = GraphProperties(is_symmetric=False, is_multigraph=False)
    G = SGGraph(
        resource_handle,
        props,
        srcs,
        dsts,
        weight_array=weights,
        edge_type_array=edge_types,
        edge_start_time_array=edge_start_times,
        edge_end_time_array=edge_end_times,
        store_transposed=False,
        renumber=False,
        do_expensive_check=False,
    )
    return G


@pytest.mark.parametrize(
    "temporal_sampling_comparison",
    [
        "strictly_increasing",
        "strictly_decreasing",
        "monotonically_increasing",
        "monotonically_decreasing",
        "last",
    ],
)
def test_homogeneous_uniform_temporal_none_times(temporal_sampling_comparison):
    rh = ResourceHandle()
    G = _build_temporal_sg(rh)
    starts = cp.asarray([1, 2], dtype=np.int32)
    fanout = np.asarray([2], dtype=np.int32)

    result = homogeneous_uniform_temporal_neighbor_sample(
        rh,
        G,
        None,
        starts,
        None,
        None,
        fanout,
        with_replacement=False,
        do_expensive_check=True,
        temporal_sampling_comparison=temporal_sampling_comparison,
    )
    result = {k: v for k, v in result.items() if v is not None}

    assert isinstance(result["majors"], cp.ndarray)
    assert isinstance(result["minors"], cp.ndarray)
    assert isinstance(result["edge_start_time"], cp.ndarray)
    assert isinstance(result["edge_end_time"], cp.ndarray)


@pytest.mark.parametrize(
    "temporal_sampling_comparison",
    [
        "strictly_increasing",
        "strictly_decreasing",
        "monotonically_increasing",
        "monotonically_decreasing",
        "last",
    ],
)
def test_homogeneous_uniform_temporal_with_times_and_labels(
    temporal_sampling_comparison,
):
    rh = ResourceHandle()
    G = _build_temporal_sg(rh)
    starts = cp.asarray([1, 2, 1], dtype=np.int32)
    start_times = cp.asarray([5, 6, 7], dtype=np.int32)
    label_offsets = cp.asarray([0, 2, 3], dtype=np.int64)
    fanout = np.asarray([1], dtype=np.int32)

    result = homogeneous_uniform_temporal_neighbor_sample(
        rh,
        G,
        None,
        starts,
        start_times,
        label_offsets,
        fanout,
        with_replacement=False,
        do_expensive_check=True,
        temporal_sampling_comparison=temporal_sampling_comparison,
    )
    result = {k: v for k, v in result.items() if v is not None}

    assert result["majors"].size == result["minors"].size
    assert result["edge_start_time"].size == result["edge_end_time"].size


@pytest.mark.parametrize(
    "temporal_sampling_comparison",
    [
        "strictly_increasing",
        "strictly_decreasing",
        "monotonically_increasing",
        "monotonically_decreasing",
        "last",
    ],
)
def test_homogeneous_biased_temporal_with_times(temporal_sampling_comparison):
    rh = ResourceHandle()
    G = _build_temporal_sg(rh)
    starts = cp.asarray([0, 1], dtype=np.int32)
    start_times = cp.asarray([0, 1], dtype=np.int32)
    fanout = np.asarray([1], dtype=np.int32)

    result = homogeneous_biased_temporal_neighbor_sample(
        rh,
        G,
        None,
        starts,
        start_times,
        None,
        fanout,
        with_replacement=False,
        do_expensive_check=True,
        temporal_sampling_comparison=temporal_sampling_comparison,
    )
    result = {k: v for k, v in result.items() if v is not None}
    assert "edge_start_time" in result and "edge_end_time" in result


@pytest.mark.parametrize(
    "temporal_sampling_comparison",
    [
        "strictly_increasing",
        "strictly_decreasing",
        "monotonically_increasing",
        "monotonically_decreasing",
        "last",
    ],
)
def test_heterogeneous_uniform_temporal_none_times(temporal_sampling_comparison):
    rh = ResourceHandle()
    G = _build_temporal_sg_with_edge_types(rh)
    starts = cp.asarray([1, 2], dtype=np.int32)
    fanout = np.asarray([1], dtype=np.int32)

    result = heterogeneous_uniform_temporal_neighbor_sample(
        rh,
        G,
        None,
        starts,
        None,
        None,
        None,
        fanout,
        num_edge_types=2,
        with_replacement=False,
        do_expensive_check=True,
        temporal_sampling_comparison=temporal_sampling_comparison,
    )
    result = {k: v for k, v in result.items() if v is not None}
    assert "edge_type" in result and "edge_start_time" in result


@pytest.mark.parametrize(
    "temporal_sampling_comparison",
    [
        "strictly_increasing",
        "strictly_decreasing",
        "monotonically_increasing",
        "monotonically_decreasing",
        "last",
    ],
)
def test_heterogeneous_biased_temporal_with_times(temporal_sampling_comparison):
    rh = ResourceHandle()
    G = _build_temporal_sg_with_edge_types(rh)
    starts = cp.asarray([0, 1], dtype=np.int32)
    start_times = cp.asarray([0, 1], dtype=np.int32)
    fanout = np.asarray([1], dtype=np.int32)

    result = heterogeneous_biased_temporal_neighbor_sample(
        rh,
        G,
        None,
        starts,
        start_times,
        None,
        None,
        fanout,
        num_edge_types=2,
        with_replacement=False,
        do_expensive_check=True,
        temporal_sampling_comparison=temporal_sampling_comparison,
    )
    result = {k: v for k, v in result.items() if v is not None}
    assert (
        "edge_type" in result
        and result["edge_start_time"].size == result["edge_end_time"].size
    )


def test_starting_vertex_times_length_mismatch_raises():
    rh = ResourceHandle()
    G = _build_temporal_sg(rh)
    starts = cp.asarray([1, 2], dtype=np.int32)
    bad_times = cp.asarray([1], dtype=np.int32)
    fanout = np.asarray([1], dtype=np.int32)

    with pytest.raises(Exception):
        homogeneous_uniform_temporal_neighbor_sample(
            rh,
            G,
            None,
            starts,
            bad_times,
            None,
            fanout,
            with_replacement=False,
            do_expensive_check=True,
        )
