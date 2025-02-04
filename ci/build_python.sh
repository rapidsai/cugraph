#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
LIBRMM_CHANNEL=$(_rapids-get-pr-artifact rmm 1808 cpp conda)
PYLIBRMM_CHANNEL=$(_rapids-get-pr-artifact rmm 1808 python conda)
LIBCUDF_CHANNEL=$(_rapids-get-pr-artifact cudf 17899 cpp conda)
PYLIBCUDF_CHANNEL=$(_rapids-get-pr-artifact cudf 17899 python conda)
LIBRAFT_CHANNEL=$(_rapids-get-pr-artifact raft 2566 cpp conda)
PYLIBRAFT_CHANNEL=$(_rapids-get-pr-artifact raft 2566 cpp python)

rapids-generate-version > ./VERSION
export RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)

rapids-logger "Begin py build"

sccache --zero-stats

# TODO: Remove `--no-test` flags once importing on a CPU
# node works correctly
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${LIBCUDF_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${PYLIBRMM_CHANNEL}" \
  --channel "${PYLIBCUDF_CHANNEL}" \
  --channel "${PYLIBRAFT_CHANNEL}" \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/pylibcugraph

sccache --show-adv-stats
sccache --zero-stats

rapids-conda-retry mambabuild \
  --no-test \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${PYLIBRMM_CHANNEL}" \
  --channel "${PYLIBRAFT_CHANNEL}" \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/cugraph

sccache --show-adv-stats

# NOTE: nothing in the cugraph-service packages are CUDA-specific, but they are
# built on each CUDA platform to ensure they are included in each set of
# artifacts, since test scripts only install from one set of artifacts based on
# the CUDA version used for the test run.
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${PYLIBRMM_CHANNEL}" \
  --channel "${PYLIBRAFT_CHANNEL}" \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/cugraph-service

rapids-upload-conda-to-s3 python
