#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Profiling script for windowed temporal sampling.

Compares:
- Standard temporal sampling (no window)
- Windowed B+C+D sampling

Run with nsys:
    nsys profile -o windowed_python python profile_windowed_sampling.py
"""

import time
import cupy as cp
import numpy as np

from pylibcugraph import (
    ResourceHandle,
    GraphProperties,
    SGGraph,
    homogeneous_uniform_temporal_neighbor_sample,
)


def create_temporal_graph(handle, n_vertices=100000, n_edges=1000000):
    """Create a random temporal graph."""
    print(f"Creating graph: {n_vertices} vertices, {n_edges} edges...")

    # Random edges
    rng = np.random.default_rng(42)
    srcs = cp.array(rng.integers(0, n_vertices, n_edges), dtype=np.int64)
    dsts = cp.array(rng.integers(0, n_vertices, n_edges), dtype=np.int64)

    # Sorted timestamps (important for B+C+D)
    edge_times = cp.array(
        np.sort(rng.integers(0, 365 * 24 * 3600, n_edges)), dtype=np.int64
    )

    graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)
    graph = SGGraph(
        handle,
        graph_props,
        srcs,
        dsts,
        edge_start_time_array=edge_times,
        store_transposed=True,
        renumber=False,
        do_expensive_check=False,
    )

    print("Graph created.")
    return graph, edge_times


def benchmark_standard(handle, graph, n_iterations=30, n_seeds=1000):
    """Benchmark standard temporal sampling (no window)."""
    print(f"\n{'=' * 60}")
    print("STANDARD TEMPORAL SAMPLING (no window)")
    print(f"{'=' * 60}")

    fanout = np.array([10, 10], dtype=np.int32)
    times = []

    for i in range(n_iterations):
        # Generate random seeds
        seeds = cp.array(np.random.randint(0, 100000, n_seeds), dtype=np.int64)
        seed_times = cp.zeros(n_seeds, dtype=np.int64)

        cp.cuda.Device().synchronize()
        start = time.perf_counter()

        result = homogeneous_uniform_temporal_neighbor_sample(
            handle,
            graph,
            None,
            seeds,
            seed_times,
            None,
            fanout,
            with_replacement=True,
            do_expensive_check=False,
        )

        cp.cuda.Device().synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        if i % 10 == 0:
            print(
                f"  Iter {i}: {elapsed:.2f} ms, {len(result.get('majors', []))} edges"
            )

    mean_time = np.mean(times[2:])  # Skip warmup
    print(f"\nMean time: {mean_time:.2f} ms")
    return mean_time


def benchmark_windowed(handle, graph, edge_times, n_iterations=30, n_seeds=1000):
    """Benchmark windowed B+C+D temporal sampling."""
    print(f"\n{'=' * 60}")
    print("WINDOWED B+C+D TEMPORAL SAMPLING")
    print(f"{'=' * 60}")

    fanout = np.array([10, 10], dtype=np.int32)
    window_size = 30 * 24 * 3600  # 30 days in seconds
    step_size = 24 * 3600  # 1 day

    max_time = int(cp.asnumpy(edge_times.max()))
    base_window_end = max_time - (n_iterations * step_size)

    times = []

    for i in range(n_iterations):
        window_end = base_window_end + i * step_size
        window_start = window_end - window_size

        # Generate random seeds
        seeds = cp.array(np.random.randint(0, 100000, n_seeds), dtype=np.int64)
        seed_times = cp.full(n_seeds, window_end, dtype=np.int64)

        cp.cuda.Device().synchronize()
        start = time.perf_counter()

        result = homogeneous_uniform_temporal_neighbor_sample(
            handle,
            graph,
            None,
            seeds,
            seed_times,
            None,
            fanout,
            with_replacement=True,
            do_expensive_check=False,
            window_start=window_start,
            window_end=window_end,
            window_time_unit="s",
        )

        cp.cuda.Device().synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        if i % 10 == 0:
            print(
                f"  Iter {i}: {elapsed:.2f} ms, {len(result.get('majors', []))} edges"
            )

    mean_time = np.mean(times[2:])  # Skip warmup
    print(f"\nMean time: {mean_time:.2f} ms")
    return mean_time


def main():
    print("=" * 60)
    print("WINDOWED TEMPORAL SAMPLING PROFILER")
    print("=" * 60)

    handle = ResourceHandle()
    graph, edge_times = create_temporal_graph(
        handle, n_vertices=100000, n_edges=1000000
    )

    # Warmup
    print("\nWarmup...")
    seeds = cp.array([0, 1, 2], dtype=np.int64)
    seed_times = cp.array([0, 0, 0], dtype=np.int64)
    fanout = np.array([2], dtype=np.int32)
    _ = homogeneous_uniform_temporal_neighbor_sample(
        handle,
        graph,
        None,
        seeds,
        seed_times,
        None,
        fanout,
        with_replacement=True,
        do_expensive_check=False,
    )

    # Benchmark
    standard_time = benchmark_standard(handle, graph)
    windowed_time = benchmark_windowed(handle, graph, edge_times)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Standard temporal: {standard_time:.2f} ms")
    print(f"Windowed B+C+D:    {windowed_time:.2f} ms")
    if windowed_time < standard_time:
        speedup = (standard_time - windowed_time) / standard_time * 100
        print(f"Improvement:       {speedup:.1f}% faster")
    else:
        slowdown = (windowed_time - standard_time) / standard_time * 100
        print(f"Slower by:         {slowdown:.1f}%")


if __name__ == "__main__":
    main()
