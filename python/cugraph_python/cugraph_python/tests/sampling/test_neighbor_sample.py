# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for uniform and temporal neighbor sampling."""

import pytest

from testing_utils import device_float32, device_int32, to_host

cp = pytest.importorskip("cupy")
cg = pytest.importorskip("cugraph_python")


def _sampled_rows(sources, destinations, hops, *properties):
    columns = [
        to_host(sources).tolist(),
        to_host(destinations).tolist(),
        to_host(hops).tolist(),
    ]
    columns.extend(to_host(prop).tolist() for prop in properties)
    return set(zip(*columns))


def test_uniform_neighbor_sample():
    ctx = cg.Context()
    graph, properties = cg.create_graph(
        cg.copy_from_arrays(
            ctx,
            device_int32([10, 10, 20]),
            device_int32([20, 30, 40]),
            device_float32([1.0, 9.0, 4.0]),
        )
    )

    rng = cg.RngState(0)
    sources, destinations, hops, sampled, offsets = cg.neighbor_sample(
        graph,
        rng,
        starting_vertices=[10],
        fan_out=[2, 1],
        edge_properties=properties,
        sampling_options=cg.SamplingOptions(return_hops=True, with_replacement=False),
    )
    assert rng.seed == 0
    assert rng.base_subsequence > 0

    assert offsets is None
    assert len(sampled) == 1
    assert _sampled_rows(sources, destinations, hops, sampled[0]) == {
        (10, 20, 0, 1.0),
        (10, 30, 0, 9.0),
        (20, 40, 1, 4.0),
    }


def test_temporal_neighbor_sample_keeps_later_edges():
    ctx = cg.Context()
    # 20 is reached at time 5, so 20 -> 40 (time 4) is too early and 20 -> 50 (time 9) remains.
    # 30 is reached at time 1, so 30 -> 60 (time 2) remains and 30 -> 70 (time 0) is too early.
    # properties[0] is the start time and properties[1] is a weight carried with the edge.
    graph, properties = cg.create_graph(
        cg.copy_from_arrays(
            ctx,
            device_int32([10, 10, 20, 20, 30, 30]),
            device_int32([20, 30, 40, 50, 60, 70]),
            edge_properties=[
                device_int32([5, 1, 4, 9, 2, 0]),
                device_float32([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
            ],
        )
    )

    sources, destinations, hops, sampled, offsets = cg.neighbor_sample(
        graph,
        cg.RngState(0),
        edge_start_times=properties[0],
        starting_vertices=[10],
        starting_vertex_start_times=[0],
        fan_out=[2, 2],
        edge_properties=properties,
        sampling_options=cg.SamplingOptions(
            return_hops=True,
            with_replacement=False,
            disjoint_sampling=True,
            temporal_sampling_comparison="strictly_increasing",
        ),
    )

    assert offsets is None
    assert len(sampled) == 2
    assert _sampled_rows(sources, destinations, hops, sampled[0], sampled[1]) == {
        (10, 20, 0, 5, 10.0),
        (10, 30, 0, 1, 20.0),
        (20, 50, 1, 9, 40.0),
        (30, 60, 1, 2, 50.0),
    }


def test_neighbor_sample_rejects_transposed_graph():
    ctx = cg.Context()
    graph, _properties = cg.create_graph(
        cg.copy_from_arrays(ctx, device_int32([10, 20]), device_int32([20, 10])),
        store_transposed=True,
    )

    with pytest.raises(ValueError, match="store_transposed"):
        cg.neighbor_sample(graph, cg.RngState(0), starting_vertices=[10], fan_out=[1])


def test_temporal_neighbor_sample_requires_start_times():
    ctx = cg.Context()
    graph, _properties = cg.create_graph(
        cg.copy_from_arrays(ctx, device_int32([10]), device_int32([20]))
    )

    with pytest.raises(ValueError, match="start_times"):
        cg.neighbor_sample(
            graph,
            cg.RngState(0),
            starting_vertices=[10],
            fan_out=[1],
            sampling_options=cg.SamplingOptions(
                temporal_sampling_comparison="strictly_increasing",
                disjoint_sampling=True,
            ),
        )
