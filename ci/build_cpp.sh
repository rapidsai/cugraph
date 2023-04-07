#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
LIBRMM_CHANNEL=$(rapids-get-artifact ci/rmm/pull-request/1223/72e0c74/rmm_conda_cpp_cuda${RAPIDS_CUDA_MAJOR}_$(arch).tar.gz)
LIBRAFT_CHANNEL=$(rapids-get-artifact ci/raft/pull-request/1388/7bddaee/raft_conda_cpp_cuda${RAPIDS_CUDA_MAJOR}_$(arch).tar.gz)
LIBCUGRAPHOPS_CHANNEL=$(rapids-get-artifact ci/cugraph-ops/pull-request/464/9ff8580/cugraph-ops_conda_cpp_cuda${RAPIDS_CUDA_MAJOR}_$(arch).tar.gz)

rapids-print-env

rapids-logger "Begin cpp build"

rapids-mamba-retry mambabuild --channel "${LIBRMM_CHANNEL}" --channel "${LIBRAFT_CHANNEL}" --channel "${LIBCUGRAPHOPS_CHANNEL}" conda/recipes/libcugraph

rapids-upload-conda-to-s3 cpp
