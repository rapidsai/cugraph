#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.
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


# location to store datasets used for benchmarking
export RAPIDS_DATASET_ROOT_DIR=${RAPIDS_DATASET_ROOT_DIR:-/datasets/cugraph}
mkdir -p logs

# list of algos, datasets, and back-ends to use in combinations
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
   amazon0302
   cit-patents
   hollywood
   soc-livejournal
"
# None backend is default networkx
# cugraph-preconvert backend is nx-cugraph
backends="
    None
    cugraph-preconverted
"

# edit this directly to for pytest
# e.g. -k "and not 100 and not 1000"
bc_k_values=""

# check for --cpu-only or --gpu-only args
if [[ "$#" -eq 1 ]]; then
    case $1 in
        --cpu-only)
            backends="None"
            ;;
        --gpu-only)
            backends="cugraph-preconverted"
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
fi

for algo in $algos; do
    for dataset in $datasets; do
        for backend in $backends; do
            name="${backend}__${algo}__${dataset}"
            echo "Running: $backend, $dataset, bench_$algo"

            # uncomment to get command for reproducing test
            # echo "RUNNING: \"pytest -sv -k \"$backend and $dataset and bench_$algo $bc_k_values\" --benchmark-json=\"logs/${name}.json\" bench_algos.py"

            pytest -sv --co \
                -k "$backend and $dataset and bench_$algo $bc_k_values" \
                --benchmark-json="logs/${name}.json" \
                bench_algos.py 2>&1 | tee "logs/${name}.out"
        done
    done
done
