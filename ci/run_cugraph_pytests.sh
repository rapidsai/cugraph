#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking run_cugraph_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cugraph/cugraph

DASK_WORKER_DEVICES="${DASK_WORKER_DEVICES:-0}" \
DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL="${DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL:-1000s}" \
DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="${DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT:-1000s}" \
DASK_CUDA_WAIT_WORKERS_MIN_TIMEOUT="${DASK_CUDA_WAIT_WORKERS_MIN_TIMEOUT:-1000s}" \
pytest --cache-clear --import-mode=append --benchmark-disable \
  -k "not test_property_graph_mg and not test_bulk_sampler_io" \
  "$@" \
  tests
