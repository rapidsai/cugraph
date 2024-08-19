#!/bin/bash
#
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Runs benchmarks for the 24.02 algos.
# Pass either a or b or both. This is useful for separating batches of runs on different GPUs:
# CUDA_VISIBLE_DEVICES=1 run-2402.sh b

export RAPIDS_DATASET_ROOT_DIR=/datasets/cugraph
mkdir -p logs

algos="
    pagerank
    betweenness_centrality
    louvain
    shortest_path
    weakly_connected_components
    triangles
    bfs_predecessors
"

datasets="
   netscience
   email_Eu_core
   cit_patents
   hollywood
   soc-livejournal
"

# None backend is default networkx
# cugraph-preconvert backend is nx-cugraph
backends="
    None
    cugraph-preconverted
"

for dataset in $datasets; do
    python ensure_dataset_accessible.py $dataset
    for backend in $backends; do
        for algo in $algos; do
            name="${backend}__${algo}__${dataset}"
            echo "RUNNING: \"pytest -sv -k \"$backend and $dataset and bench_$algo and not 1000\" --benchmark-json=\"logs/${name}.json\" bench_algos.py"
            pytest -sv -k "$backend and $dataset and bench_$algo and not 1000" --benchmark-json="logs/${name}.json" bench_algos.py 2>&1 | tee "logs/${name}.out"
        done
    done
done
