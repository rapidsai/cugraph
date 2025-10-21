#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

THIS_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

#WEIGHTED_ALGOS="--algo=bfs --algo=sssp"
#UNWEIGHTED_ALGOS="--algo=wcc"
UNWEIGHTED_ALGOS="--algo=pagerank --algo=katz"
GPU_CONFIGS="0 0,1 0,1,2,3 0,1,2,3,4,5,6,7"
SCALE_VALUES='23 24 25 26 27 28 29 30'
EDGEFACTOR_VALUES='16'

rm -f out.csv

for scale in $SCALE_VALUES; do
    echo "************************************************************"
    echo ">>>>>>>>>>  SCALE: $scale  <<<<<<<<<<"
    echo "************************************************************"
    for gpus in $GPU_CONFIGS; do
        echo ""
        echo ">>>>>>>>  CUDA_VISIBLE_DEVICES: $gpus"
        for edgefactor in $EDGEFACTOR_VALUES; do
            echo ""
            echo ">>>>>>>>>>>>>>>>> EDGEFACTOR: $edgefactor"
            #env CUDA_VISIBLE_DEVICES="$gpus" python "$THIS_SCRIPT_DIR"/main.py $WEIGHTED_ALGOS --scale=$scale --symmetric-graph
            env CUDA_VISIBLE_DEVICES="$gpus" python "$THIS_SCRIPT_DIR"/main.py "$UNWEIGHTED_ALGOS" --unweighted --symmetric-graph --scale="$scale" --edgefactor="$edgefactor"
        done
    done
    mv out.csv random_scale_"$scale".csv
done
