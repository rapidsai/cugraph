#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PY_VER=${RAPIDS_PY_VERSION//./}
LIBRAFT_CHANNEL=$(rapids-get-artifact ci/raft/pull-request/1333/c60c0cb/raft_conda_cpp_cuda11_$(arch).tar.gz)
RAFT_CHANNEL=$(rapids-get-artifact ci/raft/pull-request/1333/c60c0cb/raft_conda_python_cuda11_${PY_VER}_$(arch).tar.gz)

rapids-logger "Begin py build"

# TODO: Remove `--no-test` flags once importing on a CPU
# node works correctly
rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${RAFT_CHANNEL}" \
  conda/recipes/pylibcugraph

rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${RAFT_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/cugraph

rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${RAFT_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/cugraph-service

rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${RAFT_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/cugraph-pyg

rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${RAFT_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  --channel dglteam \
  --channel pytorch \
  conda/recipes/cugraph-dgl

rapids-upload-conda-to-s3 python
