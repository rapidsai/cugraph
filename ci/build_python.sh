#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-logger "Begin py build"

# TODO: Remove `--no-test` flags once importing on a CPU
# node works correctly
rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/pylibcugraph

rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/cugraph

# NOTE: nothing in cugraph-nx is CUDA-specific, but it is built on each CUDA
# platform to ensure it is included in each set of artifacts since tests only
# install from one set of CUDA-specific artifacts.
rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/cugraph-nx

# NOTE: nothing in the cugraph-service packages are CUDA-specific, but they are
# built on each CUDA platform to ensure they areincluded in each set of
# artifacts since tests only install from one set of CUDA-specific artifacts.
rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/cugraph-service

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"

if [[ ${RAPIDS_CUDA_MAJOR} == "11" ]]; then
  # Only CUDA 11 is supported right now due to PyTorch requirement.
  rapids-mamba-retry mambabuild \
    --no-test \
    --channel "${CPP_CHANNEL}" \
    --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
    --channel pyg \
    --channel pytorch \
    --channel pytorch-nightly \
    conda/recipes/cugraph-pyg

  # Only CUDA 11 is supported right now due to PyTorch requirement.
  rapids-mamba-retry mambabuild \
    --no-test \
    --channel "${CPP_CHANNEL}" \
    --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
    --channel dglteam \
    --channel pytorch \
    --channel pytorch-nightly \
    conda/recipes/cugraph-dgl
fi

rapids-upload-conda-to-s3 python
