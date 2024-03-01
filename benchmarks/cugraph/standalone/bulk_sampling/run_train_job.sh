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
REPLICATION_FACTOR=2
JOB_ID=$RANDOM

# options: PyG, cuGraphPyG, or cuGraphDGL
FRAMEWORK="cuGraphDGL"
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
        bash /scripts/run_train.sh $BATCH_SIZE $FANOUT $REPLICATION_FACTOR "/scripts" $NUM_EPOCHS "cugraph_pyg" $nnodes $head_node_ip $JOB_ID
elif [[ "$FRAMEWORK" == "cuGraphDGL" ]]; then
    srun \
        --container-image $CONTAINER_IMAGE \
        --container-mounts=${LOGS_DIR}":/logs",${SAMPLES_DIR}":/samples",${SCRIPTS_DIR}":/scripts",${DATASETS_DIR}":/datasets" \
        bash /scripts/run_train.sh $BATCH_SIZE $FANOUT $REPLICATION_FACTOR "/scripts" $NUM_EPOCHS "cugraph_dgl_csr" $nnodes $head_node_ip $JOB_ID
fi

