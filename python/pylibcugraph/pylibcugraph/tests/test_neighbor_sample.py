# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import cupy as cp
import pytest

from pylibcugraph import (
    ResourceHandle,
    GraphProperties,
    SGGraph,
    homogeneous_uniform_neighbor_sample,
    homogeneous_biased_neighbor_sample,
    heterogeneous_uniform_neighbor_sample,
    heterogeneous_biased_neighbor_sample,
)


def _check_edges(result_srcs, result_dsts, result_props, srcs, dsts, props, num_verts):
    h_src_arr = srcs
    h_dst_arr = dsts
    h_prop_arr = props

    if isinstance(h_src_arr, cp.ndarray):
        h_src_arr = h_src_arr.get()
    if isinstance(h_dst_arr, cp.ndarray):
        h_dst_arr = h_dst_arr.get()
    if isinstance(h_prop_arr, cp.ndarray):
        h_prop_arr = h_prop_arr.get()

    h_result_srcs = result_srcs.get()
    h_result_dsts = result_dsts.get()
    h_result_props = result_props.get()

    # Following the C validation, we will check that all edges are part of the
    # graph
    M = np.zeros((num_verts, num_verts), dtype=props.dtype)

    # Construct the adjacency matrix
    for idx in range(len(h_src_arr)):
        M[h_dst_arr[idx]][h_src_arr[idx]] = h_prop_arr[idx]

    for edge in range(len(h_result_srcs)):
        assert M[h_result_dsts[edge]][h_result_srcs[edge]] == h_result_props[edge]


def _build_sg(
    resource_handle: ResourceHandle,
    with_edge_types: bool = False,
    with_edge_ids: bool = False,
) -> SGGraph:
    srcs = cp.asarray([0, 1, 1, 2], dtype=np.int32)
    dsts = cp.asarray([1, 2, 3, 3], dtype=np.int32)
    weights = cp.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    edge_types = cp.asarray([0, 1, 0, 1], dtype=np.int32) if (with_edge_types) else None
    edge_ids = cp.asarray([1, 2, 3, 4], dtype=np.int32) if (with_edge_ids) else None

    props = GraphProperties(is_symmetric=False, is_multigraph=False)

    G = SGGraph(
        resource_handle,
        props,
        srcs,
        dsts,
        weight_array=weights,
        edge_type_array=edge_types,
        edge_id_array=edge_ids,
        store_transposed=False,
        renumber=False,
        do_expensive_check=False,
    )

    return (G, srcs, dsts, weights, edge_types, edge_ids, 4)


def test_homogeneous_uniform_neighbor_sample_none_labels():
    rh = ResourceHandle()
    G, srcs, dsts, weights, edge_types, edge_ids, num_verts = _build_sg(rh)
    starts = cp.asarray([1, 2], dtype=np.int32)
    fanout = np.asarray([2], dtype=np.int32)

    result = homogeneous_uniform_neighbor_sample(
        rh,
        G,
        starts,
        None,
        fanout,
        with_replacement=False,
        do_expensive_check=True,
    )
    result = {k: v for k, v in result.items() if v is not None}

    assert isinstance(result["majors"], cp.ndarray)
    assert isinstance(result["minors"], cp.ndarray)
    assert result["majors"].size == result["minors"].size
    print("result", result)
    _check_edges(
        result["majors"],
        result["minors"],
        result["weight"],
        srcs,
        dsts,
        weights,
        num_verts,
    )


def test_homogeneous_uniform_neighbor_sample_with_labels():
    rh = ResourceHandle()
    G, srcs, dsts, weights, edge_types, edge_ids, num_verts = _build_sg(rh)
    starts = cp.asarray([1, 2, 1], dtype=np.int32)
    label_offsets = cp.asarray([0, 2, 3], dtype=np.int64)
    fanout = np.asarray([1], dtype=np.int32)

    result = homogeneous_uniform_neighbor_sample(
        rh,
        G,
        starts,
        label_offsets,
        fanout,
        with_replacement=False,
        do_expensive_check=True,
    )
    result = {k: v for k, v in result.items() if v is not None}

    assert result["majors"].size == result["minors"].size
    print("result", result)
    _check_edges(
        result["majors"],
        result["minors"],
        result["weight"],
        srcs,
        dsts,
        weights,
        num_verts,
    )


def test_homogeneous_biased_neighbor_sample_basic():
    rh = ResourceHandle()
    G, srcs, dsts, weights, edge_types, edge_ids, num_verts = _build_sg(rh)
    starts = cp.asarray([0, 1], dtype=np.int32)
    fanout = np.asarray([1], dtype=np.int32)

    result = homogeneous_biased_neighbor_sample(
        rh,
        G,
        starts,
        None,
        fanout,
        with_replacement=False,
        do_expensive_check=True,
    )
    result = {k: v for k, v in result.items() if v is not None}

    assert "majors" in result and "minors" in result
    assert result["majors"].size == result["minors"].size
    print("result", result)
    _check_edges(
        result["majors"],
        result["minors"],
        result["weight"],
        srcs,
        dsts,
        weights,
        num_verts,
    )


def test_heterogeneous_uniform_neighbor_sample_basic():
    rh = ResourceHandle()
    G, srcs, dsts, weights, edge_types, edge_ids, num_verts = _build_sg(rh, True)
    starts = cp.asarray([1, 2], dtype=np.int32)
    fanout = np.asarray([1], dtype=np.int32)

    result = heterogeneous_uniform_neighbor_sample(
        rh,
        G,
        starts,
        None,
        None,
        fanout,
        num_edge_types=2,
        with_replacement=False,
        do_expensive_check=True,
    )
    result = {k: v for k, v in result.items() if v is not None}

    assert "edge_type" in result
    assert result["majors"].size == result["minors"].size
    print("result", result)
    _check_edges(
        result["majors"],
        result["minors"],
        result["weight"],
        srcs,
        dsts,
        weights,
        num_verts,
    )
    _check_edges(
        result["majors"],
        result["minors"],
        result["edge_type"],
        srcs,
        dsts,
        edge_types,
        num_verts,
    )
    if edge_ids:
        _check_edges(
            result["majors"],
            result["minors"],
            result["edge_ids"],
            srcs,
            dsts,
            edge_ids,
            num_verts,
        )


def test_heterogeneous_biased_neighbor_sample_basic():
    rh = ResourceHandle()
    G, srcs, dsts, weights, edge_types, edge_ids, num_verts = _build_sg(rh, True)
    starts = cp.asarray([0, 1], dtype=np.int32)
    fanout = np.asarray([1], dtype=np.int32)

    result = heterogeneous_biased_neighbor_sample(
        rh,
        G,
        starts,
        None,
        None,
        fanout,
        num_edge_types=2,
        with_replacement=False,
        do_expensive_check=True,
    )
    result = {k: v for k, v in result.items() if v is not None}

    assert "edge_type" in result
    assert result["majors"].size == result["minors"].size
    print("result", result)
    _check_edges(
        result["majors"],
        result["minors"],
        result["weight"],
        srcs,
        dsts,
        weights,
        num_verts,
    )
    _check_edges(
        result["majors"],
        result["minors"],
        result["edge_type"],
        srcs,
        dsts,
        edge_types,
        num_verts,
    )
    if edge_ids:
        _check_edges(
            result["majors"],
            result["minors"],
            result["edge_ids"],
            srcs,
            dsts,
            edge_ids,
            num_verts,
        )


def test_starting_vertex_label_offsets_length_mismatch_raises():
    rh = ResourceHandle()
    G, srcs, dsts, weights, edge_types, edge_ids, num_verts = _build_sg(rh)
    starts = cp.asarray([1, 2], dtype=np.int32)
    bad_offsets = cp.asarray([0, 1], dtype=np.int64)  # last != len(starts)
    fanout = np.asarray([1], dtype=np.int32)

    with pytest.raises(Exception):
        homogeneous_uniform_neighbor_sample(
            rh,
            G,
            starts,
            bad_offsets,
            fanout,
            with_replacement=False,
            do_expensive_check=True,
        )
