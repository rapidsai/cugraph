#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

################################################################################
NUMARGS=$#
ARGS=$*
VALIDARGS="-h --help scheduler workers --tcp --ucx --ucxib --ucx-ib"
HELP="$0 [<app> ...] [<flag> ...]
 where <app> is:
   scheduler               - start dask scheduler
   workers                 - start dask workers
 and <flag> is:
   --tcp                   - initalize a TCP cluster (default)
   --ucx                   - initialize a UCX cluster with NVLink
   --ucxib | --ucx-ib      - initialize a UCX cluster with InfiniBand+NVLink
   -h | --help             - print this text

 The cluster config order of precedence is any specification on the
 command line (--tcp, --ucx, etc.) if provided, then the value of the
 env var DASK_CLUSTER_CONFIG_TYPE if set, then the default value of TCP.

 The env var SCHEDULER_FILE must be set to the location of the dask scheduler
 file that the scheduler will generate and the worker(s) will read. This
 location must be accessible by the scheduler and workers, meaning a multi-node
 configuration will need to set this to a location on a shared file system.
"

# Default configuration variables. Most are defined using the bash := or :-
# syntax, which means they will be set only if they were previously unset in
# the environment.
WORKER_RMM_POOL_SIZE=${WORKER_RMM_POOL_SIZE:-12G}
DASK_CUDA_INTERFACE=${DASK_CUDA_INTERFACE:-ibp5s0f0}
DASK_SCHEDULER_PORT=${DASK_SCHEDULER_PORT:-8792}
DASK_DEVICE_MEMORY_LIMIT=${DASK_DEVICE_MEMORY_LIMIT:-auto}
DASK_HOST_MEMORY_LIMIT=${DASK_HOST_MEMORY_LIMIT:-auto}

# Logs can be written to a specific location by setting the DASK_LOGS_DIR
# env var. If unset, all logs are created under a dir named after the
# current PID.
DASK_LOGS_DIR=${DASK_LOGS_DIR:-dask_logs-$$}
DASK_SCHEDULER_LOG=${DASK_LOGS_DIR}/scheduler_log.txt
DASK_WORKERS_LOG=${DASK_LOGS_DIR}/worker-${HOSTNAME}_log.txt

# DASK_CLUSTER_CONFIG_TYPE defaults to the env var value if set, else TCP. CLI
# options to this script take precedence. Valid values are TCP, UCX, UCXIB
DASK_CLUSTER_CONFIG_TYPE=${DASK_CLUSTER_CONFIG_TYPE:-TCP}


################################################################################
# FUNCTIONS

hasArg () {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

logger_prefix=">>>> "
logger () {
    if (( $# > 0 )) && [ "$1" == "-p" ]; then
        shift
        echo -e "${logger_prefix}$*"
    else
        echo -e "$(date --utc "+%D-%T.%N")_UTC${logger_prefix}$*"
    fi
}

buildTcpArgs () {
    export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="100s"
    export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP="600s"
    export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MIN="1s"
    export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MAX="60s"
    export DASK_DISTRIBUTED__WORKER__MEMORY__Terminate="False"

    SCHEDULER_ARGS="--protocol=tcp
                    --port=$DASK_SCHEDULER_PORT
                    --scheduler-file $SCHEDULER_FILE
                "

    WORKER_ARGS="--rmm-pool-size=$WORKER_RMM_POOL_SIZE
             --local-directory=/tmp/$LOGNAME
             --scheduler-file=$SCHEDULER_FILE
             --memory-limit=$DASK_HOST_MEMORY_LIMIT
             --device-memory-limit=$DASK_DEVICE_MEMORY_LIMIT
            "

}

buildUCXWithInfinibandArgs () {
    export DASK_RMM__POOL_SIZE=0.5GB
    export DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT=True

    SCHEDULER_ARGS="--protocol=ucx
                --port=$DASK_SCHEDULER_PORT
                --interface=$DASK_CUDA_INTERFACE
                --scheduler-file $SCHEDULER_FILE
               "

    WORKER_ARGS="--interface=$DASK_CUDA_INTERFACE
                --rmm-pool-size=$WORKER_RMM_POOL_SIZE
                --rmm-async
                --local-directory=/tmp/$LOGNAME
                --scheduler-file=$SCHEDULER_FILE
                --memory-limit=$DASK_HOST_MEMORY_LIMIT
                --device-memory-limit=$DASK_DEVICE_MEMORY_LIMIT
                "
}

buildUCXwithoutInfinibandArgs () {
    export UCX_TCP_CM_REUSEADDR=y
    export UCX_MAX_RNDV_RAILS=1
    export UCX_TCP_TX_SEG_SIZE=8M
    export UCX_TCP_RX_SEG_SIZE=8M

    export DASK_DISTRIBUTED__COMM__UCX__CUDA_COPY=True
    export DASK_DISTRIBUTED__COMM__UCX__TCP=True
    export DASK_DISTRIBUTED__COMM__UCX__NVLINK=True
    export DASK_DISTRIBUTED__COMM__UCX__INFINIBAND=False
    export DASK_DISTRIBUTED__COMM__UCX__RDMACM=False
    export DASK_RMM__POOL_SIZE=0.5GB


    SCHEDULER_ARGS="--protocol=ucx
            --port=$DASK_SCHEDULER_PORT
            --scheduler-file $SCHEDULER_FILE
            "

    WORKER_ARGS="--enable-tcp-over-ucx
                --enable-nvlink
                --disable-infiniband
                --disable-rdmacm
                --rmm-pool-size=$WORKER_RMM_POOL_SIZE
                --local-directory=/tmp/$LOGNAME
                --scheduler-file=$SCHEDULER_FILE
                --memory-limit=$DASK_HOST_MEMORY_LIMIT
                --device-memory-limit=$DASK_DEVICE_MEMORY_LIMIT
                "
}

scheduler_pid=""
worker_pid=""
num_scheduler_tries=0

startScheduler () {
    mkdir -p "$(dirname "$SCHEDULER_FILE")"
    echo "RUNNING: \"dask scheduler $SCHEDULER_ARGS\"" > "$DASK_SCHEDULER_LOG"
    dask scheduler "$SCHEDULER_ARGS" >> "$DASK_SCHEDULER_LOG" 2>&1 &
    scheduler_pid=$!
}


################################################################################
# READ CLI OPTIONS

START_SCHEDULER=0
START_WORKERS=0

if (( NUMARGS == 0 )); then
    echo "${HELP}"
    exit 0
else
    if hasArg -h || hasArg --help; then
        echo "${HELP}"
        exit 0
    fi
    for a in ${ARGS}; do
        if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
            echo "Invalid option: ${a}"
            exit 1
        fi
    done
fi

if [ -z ${SCHEDULER_FILE+x} ]; then
    echo "Env var SCHEDULER_FILE must be set. See -h for details"
    exit 1
fi

if hasArg scheduler; then
    START_SCHEDULER=1
fi
if hasArg workers; then
    START_WORKERS=1
fi
# Allow the command line to take precedence
if hasArg --tcp; then
    DASK_CLUSTER_CONFIG_TYPE=TCP
elif hasArg --ucx; then
    DASK_CLUSTER_CONFIG_TYPE=UCX
elif hasArg --ucxib || hasArg --ucx-ib; then
    DASK_CLUSTER_CONFIG_TYPE=UCXIB
fi


################################################################################
# SETUP & RUN

#export DASK_LOGGING__DISTRIBUTED="DEBUG"
#ulimit -n 100000

if [[ "$DASK_CLUSTER_CONFIG_TYPE" == "UCX" ]]; then
    logger "Using cluster configurtion for UCX"
    buildUCXwithoutInfinibandArgs
elif [[ "$DASK_CLUSTER_CONFIG_TYPE" == "UCXIB" ]]; then
    logger "Using cluster configurtion for UCX with Infiniband"
    buildUCXWithInfinibandArgs
else
    logger "Using cluster configurtion for TCP"
    buildTcpArgs
fi

mkdir -p "$DASK_LOGS_DIR"
logger "Logs written to: $DASK_LOGS_DIR"

if [[ $START_SCHEDULER == 1 ]]; then
    rm -f "$SCHEDULER_FILE" "$DASK_SCHEDULER_LOG" "$DASK_WORKERS_LOG"

    startScheduler
    sleep 6
    num_scheduler_tries=$(( num_scheduler_tries+1 ))

    # Wait for the scheduler to start first before proceeding, since
    # it may require several retries (if prior run left ports open
    # that need time to close, etc.)
    while [ ! -f "$SCHEDULER_FILE" ]; do
        scheduler_alive=$(ps -p $scheduler_pid > /dev/null ; echo $?)
        if [[ $scheduler_alive != 0 ]]; then
            if [[ $num_scheduler_tries != 30 ]]; then
                logger "scheduler failed to start, retry #$num_scheduler_tries"
                startScheduler
                sleep 6
                num_scheduler_tries=$(( num_scheduler_tries+1 ))
            else
                logger "could not start scheduler, exiting."
                exit 1
            fi
        fi
    done
    logger "scheduler started."
fi

if [[ $START_WORKERS == 1 ]]; then
    rm -f "$DASK_WORKERS_LOG"
    while [ ! -f "$SCHEDULER_FILE" ]; do
        logger "run-dask-process.sh: $SCHEDULER_FILE not present - waiting to start workers..."
        sleep 2
    done
    echo "RUNNING: \"dask_cuda_worker $WORKER_ARGS\"" > "$DASK_WORKERS_LOG"
    dask-cuda-worker "$WORKER_ARGS" >> "$DASK_WORKERS_LOG" 2>&1 &
    worker_pid=$!
    logger "worker(s) started."
fi

# This script will not return until the following background process
# have been completed/killed.
if [[ $worker_pid != "" ]]; then
    logger "waiting for worker pid $worker_pid to finish before exiting script..."
    wait $worker_pid
fi
if [[ $scheduler_pid != "" ]]; then
    logger "waiting for scheduler pid $scheduler_pid to finish before exiting script..."
    wait $scheduler_pid
fi
