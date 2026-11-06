# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pylibcugraph
from pylibcugraph import GraphProperties, ResourceHandle, SGGraph


def _canonicalize_cycle(cycle_vertices):
    cycle = list(cycle_vertices)
    if len(cycle) <= 1:
        return tuple(cycle)
    min_index = cycle.index(min(cycle))
    rotated = cycle[min_index:] + cycle[:min_index]
    return tuple(rotated)


def _cycles_from_flat(cycle_vertices, cycle_offsets):
    cycles = []
    for i in range(len(cycle_offsets) - 1):
        begin = int(cycle_offsets[i])
        end = int(cycle_offsets[i + 1])
        cycles.append(_canonicalize_cycle(cycle_vertices[begin:end].tolist()))
    return sorted(cycles)


def test_simple_cycles_small_directed_graph():
    srcs = cp.asarray([0, 1, 1, 2, 2, 3], dtype=np.int32)
    dsts = cp.asarray([1, 2, 3, 0, 1, 2], dtype=np.int32)
    weights = cp.ones(6, dtype=np.float32)

    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)
    graph = SGGraph(
        resource_handle,
        graph_props,
        srcs,
        dsts,
        weight_array=weights,
        store_transposed=False,
        renumber=False,
        do_expensive_check=False,
    )

    cycle_vertices, cycle_offsets, num_cycles = pylibcugraph.simple_cycles(
        resource_handle, graph, None, 4, False
    )

    assert num_cycles == 4
    expected = sorted(
        [
            (0, 1, 2),
            (0, 1, 3, 2),
            (1, 2),
            (1, 3, 2),
        ]
    )
    assert _cycles_from_flat(cycle_vertices, cycle_offsets) == expected


def test_simple_cycles_seed_filter():
    srcs = cp.asarray([0, 1, 1, 2, 2, 3], dtype=np.int32)
    dsts = cp.asarray([1, 2, 3, 0, 1, 2], dtype=np.int32)
    weights = cp.ones(6, dtype=np.float32)

    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)
    graph = SGGraph(
        resource_handle,
        graph_props,
        srcs,
        dsts,
        weight_array=weights,
        store_transposed=False,
        renumber=False,
        do_expensive_check=False,
    )

    seeds = cp.asarray([3], dtype=np.int32)
    cycle_vertices, cycle_offsets, num_cycles = pylibcugraph.simple_cycles(
        resource_handle, graph, seeds, 4, False
    )

    assert num_cycles == 2
    expected = sorted([(0, 1, 3, 2), (1, 3, 2)])
    assert _cycles_from_flat(cycle_vertices, cycle_offsets) == expected
