#!/bin/bash
#
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Runs benchmarks for the 24.02 algos.
# Pass either a or b or both. This is useful for separating batches of runs on different GPUs:
# CUDA_VISIBLE_DEVICES=1 run-2402.sh b

mkdir -p logs

# benches="$benches ..." pattern is easy to comment out individual runs
benches=

while [[ $1 != "" ]]; do
    if [[ $1 == "a" ]]; then
        benches="$benches bench_ancestors"
        benches="$benches bench_average_clustering"
        benches="$benches bench_generic_bfs_edges"
        benches="$benches bench_bfs_edges"
        benches="$benches bench_bfs_layers"
        benches="$benches bench_bfs_predecessors"
        benches="$benches bench_bfs_successors"
        benches="$benches bench_bfs_tree"
        benches="$benches bench_clustering"
        benches="$benches bench_core_number"
        benches="$benches bench_descendants"
    elif [[ $1 == "b" ]]; then
        benches="$benches bench_descendants_at_distance"
        benches="$benches bench_is_bipartite"
        benches="$benches bench_is_strongly_connected"
        benches="$benches bench_is_weakly_connected"
        benches="$benches bench_number_strongly_connected_components"
        benches="$benches bench_number_weakly_connected_components"
        benches="$benches bench_overall_reciprocity"
        benches="$benches bench_reciprocity"
        benches="$benches bench_strongly_connected_components"
        benches="$benches bench_transitivity"
        benches="$benches bench_triangles"
        benches="$benches bench_weakly_connected_components"
    fi
    shift
done

for bench in $benches; do
    pytest -sv -k "soc-livejournal1" "bench_algos.py::$bench" 2>&1 | tee "logs/${bench}.log"
done
