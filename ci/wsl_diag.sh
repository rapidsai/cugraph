#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TEMP: WSL diagnostics for MG dask fixture failure and SG slowdown.
set -x
cat /proc/version
nproc
free -g
nvidia-smi
cat /etc/os-release | head -3

RAPIDS_DATASET_ROOT_DIR="$(pwd)/datasets"
export RAPIDS_DATASET_ROOT_DIR
export PY_IGNORE_IMPORTMISMATCH=1
export DASK_WORKER_DEVICES="0"
export DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL="1000s"
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="1000s"
export DASK_CUDA_WAIT_WORKERS_MIN_TIMEOUT="1000s"
T=./python/cugraph/cugraph/tests
PYTEST=(python -m pytest -v --import-mode=append --benchmark-disable --durations=15)

echo "=== 1. SG connectivity alone"
timeout 1200 "${PYTEST[@]}" "$T/components/test_connectivity.py"

echo "=== 2. MG fixture with NCCL_DEBUG=INFO"
NCCL_DEBUG=INFO timeout 900 "${PYTEST[@]}" -x --tb=long -s "$T/link_analysis/test_pagerank_mg.py::test_pagerank_non_convergence"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo "=== 3. MG fixture with NCCL_CUMEM_ENABLE=0"
NCCL_CUMEM_ENABLE=0 NCCL_DEBUG=INFO timeout 900 "${PYTEST[@]}" -x --tb=long -s "$T/link_analysis/test_pagerank_mg.py::test_pagerank_non_convergence"

echo "=== 4. MG fixture then SG connectivity in one session"
timeout 1800 "${PYTEST[@]}" --tb=line "$T/link_analysis/test_pagerank_mg.py::test_pagerank_non_convergence" "$T/components/test_connectivity.py"
ps aux | grep -iE "dask|python" | grep -v grep | head -20
nvidia-smi
exit 0
