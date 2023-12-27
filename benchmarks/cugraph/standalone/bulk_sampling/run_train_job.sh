#!/bin/bash

#SBATCH -A datascience_rapids_cugraphgnn
#SBATCH -p luna
#SBATCH -J datascience_rapids_cugraphgnn-papers:bulkSamplingPyG
#SBATCH -N 1
#SBATCH -t 00:22:00 

export CONTAINER_IMAGE="/lustre/fsw/rapids/abarghi/dlfw_patched.squash"
export SCRIPTS_DIR=$(pwd)
export LOGS_DIR="/lustre/fsw/rapids/abarghi/logs"
export SAMPLES_DIR="/lustre/fsw/rapids/abarghi/samples"
export DATASETS_DIR="/lustre/fsw/rapids/gnn_datasets"

export BATCH_SIZE=512
export FANOUT="10_10_10"
export REPLICATION_FACTOR=1
export NUM_EPOCHS=1
# options: PyG or cuGraphPyG
export FRAMEWORK="cuGraphPyG"

export RAPIDS_NO_INITIALIZE=1
export CUDF_SPILL=1
export LIBCUDF_CUFILE_POLICY="KVIKIO"
export KVIKIO_NTHREADS=64
export GPUS_PER_NODE=8

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

