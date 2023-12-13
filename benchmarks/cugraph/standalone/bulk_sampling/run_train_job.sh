#!/bin/bash

#SBATCH -A datascience_rapids_workflows
#SBATCH -p batch_short_dgx1_m2 
#SBATCH -J papers:trainingPyG 
#SBATCH -N 2 
#SBATCH --gpus-per-node 8 
#SBATCH -t 00:18:00 
#SBATCH --nv-meta ml-model.rapids-nightlies 
#SBATCH --exclusive 

export CONTAINER_IMAGE="/gpfs/fs1/projects/sw_rapids/users/abarghi/dlfw_12_5_23.squash"
export SCRIPTS_DIR=${pwd}
export LOGS_DIR="/gpfs/fs1/projects/sw_rapids/users/abarghi/logs"
export SAMPLES_DIR="/gpfs/fs1/projects/sw_rapids/users/abarghi/samples"
export DATASETS_DIR="/gpfs/fs1/projects/sw_rapids/users/abarghi/datasets"

export BATCH_SIZE=512
export FANOUT="10, 10, 10"
export REPLICATION_FACTOR=1

export RAPIDS_NO_INITIALIZE=1
export CUDF_SPILL=1
export LIBCUDF_CUFILE_POLICY="KVIKIO"
export KVIKIO_NTHREADS=64

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

nnodes=$SLURM_JOB_NUM_NODES
echo Num Nodes: $nnodes

gpus_per_node=$SLURM_GPUS_PER_NODE
echo Num GPUs Per Node: $gpus_per_node

# Generate samples
srun \
    --container-image $CONTAINER_IMAGE \
    --container-mounts=${LOGS_DIR}":/logs",${SAMPLES_DIR}":/samples",${SCRIPTS_DIR}":/scripts",${DATASETS_DIR}":/datasets" \
    bash /project/run_sampling.sh $BATCH_SIZE $FANOUT $REPLICATION_FACTOR "/scripts"

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
        /project/bench_cugraph_pyg.py \
            --output_file "/logs/output.txt" \
            --framework "cuGraph" \
            --dataset_dir "/datasets" \
            --sample_dir "/samples/ogbn_papers100M["$REPLICATION_FACTOR"]_b"$BATCH_SIZE"_f["$FANOUT"]" \
            --replication_factor $REPLICATION_FACTOR

