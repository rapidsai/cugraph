#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

REPO="rmm"
PR_NUMBER="1095"
COMMIT=$(git ls-remote https://github.com/rapidsai/${REPO}.git refs/heads/pull-request/${PR_NUMBER} | cut -c1-7)
RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
PYTHON_MINOR_VERSION=$(python --version | sed -E 's/Python [0-9]+\.([0-9]+)\.[0-9]+/\1/g')
LIBRMM_CHANNEL=$(rapids-get-artifact ci/${REPO}/pull-request/${PR_NUMBER}/${COMMIT}/rmm_conda_cpp_cuda${RAPIDS_CUDA_MAJOR}_$(arch).tar.gz)
RMM_CHANNEL=$(rapids-get-artifact ci/${REPO}/pull-request/${PR_NUMBER}/${COMMIT}/rmm_conda_python_cuda${RAPIDS_CUDA_MAJOR}_3${PYTHON_MINOR_VERSION}_$(arch).tar.gz)

rapids-print-env

version=$(rapids-generate-version)

rapids-logger "Begin cpp build"

RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  conda/recipes/libcugraph

rapids-upload-conda-to-s3 cpp
