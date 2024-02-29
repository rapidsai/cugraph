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

#SBATCH -A datascience_rapids_cugraphgnn
#SBATCH -p luna
#SBATCH -J datascience_rapids_cugraphgnn-papers:bulkSamplingPyG
#SBATCH -N 1
#SBATCH -t 00:25:00

CONTAINER_IMAGE=${CONTAINER_IMAGE:="please_specify_container"}
SCRIPTS_DIR=$(pwd)
LOGS_DIR=${LOGS_DIR:=$(pwd)"/logs"}
SAMPLES_DIR=${SAMPLES_DIR:=$(pwd)/samples}
DATASETS_DIR=${DATASETS_DIR:=$(pwd)/datasets}

mkdir -p $LOGS_DIR
mkdir -p $SAMPLES_DIR
mkdir -p $DATASETS_DIR

BATCH_SIZE=512
FANOUT="10_10_10"
NUM_EPOCHS=1
REPLICATION_FACTOR=1

# options: PyG or cuGraphPyG
FRAMEWORK="cuGraphPyG"
GPUS_PER_NODE=8

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

nnodes=$SLURM_JOB_NUM_NODES
echo Num Nodes: $nnodes

gpus_per_node=$GPUS_PER_NODE
echo Num GPUs Per Node: $gpus_per_node

set -e

# First run without cuGraph to get data

if [[ "$FRAMEWORK" == "cuGraphPyG" ]]; then
    # Generate samples
    srun \
        --container-image $CONTAINER_IMAGE \
        --container-mounts=${LOGS_DIR}":/logs",${SAMPLES_DIR}":/samples",${SCRIPTS_DIR}":/scripts",${DATASETS_DIR}":/datasets" \
        bash /scripts/run_sampling.sh $BATCH_SIZE $FANOUT $REPLICATION_FACTOR "/scripts" $NUM_EPOCHS
fi

# Train
srun \
    --container-image $CONTAINER_IMAGE \
    --container-mounts=${LOGS_DIR}":/logs",${SAMPLES_DIR}":/samples",${SCRIPTS_DIR}":/scripts",${DATASETS_DIR}":/datasets" \
    torchrun \
        --nnodes $nnodes \
        --nproc-per-node $gpus_per_node \
        --rdzv-id $RANDOM \
        --rdzv-backend c10d \
        --rdzv-endpoint $head_node_ip:29500 \
        /scripts/bench_cugraph_training.py \
            --output_file "/logs/output.txt" \
            --framework $FRAMEWORK \
            --dataset_dir "/datasets" \
            --sample_dir "/samples" \
            --batch_size $BATCH_SIZE \
            --fanout $FANOUT \
            --replication_factor $REPLICATION_FACTOR \
            --num_epochs $NUM_EPOCHS
