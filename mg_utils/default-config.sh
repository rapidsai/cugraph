#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Most are defined using the bash := or :- syntax, which means they
# will be set only if they were previously unset. The project config
# is loaded first, which gives it the opportunity to override anything
# in this file that uses that syntax.  If there are variables in this
# file that should not be overridded by a project, then they will
# simply not use that syntax and override, since these variables are
# read last.
WORKSPACE=$THIS_DIR

# These really should be oerridden by the project config!
CONDA_ENV=${CONDA_ENV:-rapids}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
WORKER_RMM_POOL_SIZE=${WORKER_RMM_POOL_SIZE:-12G}
DASK_CUDA_INTERFACE=${DASK_CUDA_INTERFACE:-ib0}
DASK_SCHEDULER_PORT=${DASK_SCHEDULER_PORT:-8792}
DASK_DEVICE_MEMORY_LIMIT=${DASK_DEVICE_MEMORY_LIMIT:-auto}
DASK_HOST_MEMORY_LIMIT=${DASK_HOST_MEMORY_LIMIT:-auto}

BUILD_LOG_FILE=${BUILD_LOG_FILE:-${RESULTS_DIR}/build_log.txt}
SCHEDULER_FILE=${SCHEDULER_FILE:-${WORKSPACE}/dask-scheduler.json}
DATE=${DATE:-$(date --utc "+%Y-%m-%d_%H:%M:%S")_UTC}
ENV_EXPORT_FILE=${ENV_EXPORT_FILE:-${WORKSPACE}/$(basename "${CONDA_ENV}")-${DATE}.txt}
