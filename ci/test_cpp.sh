#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking test_cpp.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

. /opt/conda/etc/profile.d/conda.sh

RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
LIBRMM_CHANNEL=$(_rapids-get-pr-artifact rmm 1808 cpp conda)
PYLIBRMM_CHANNEL=$(_rapids-get-pr-artifact rmm 1808 python conda)
LIBCUDF_CHANNEL=$(_rapids-get-pr-artifact cudf 17899 cpp conda)
PYLIBCUDF_CHANNEL=$(_rapids-get-pr-artifact cudf 17899 python conda)
LIBRAFT_CHANNEL=$(_rapids-get-pr-artifact raft 2566 cpp conda)
PYLIBRAFT_CHANNEL=$(_rapids-get-pr-artifact raft 2566 python conda)

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-print-env

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${LIBCUDF_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${PYLIBRMM_CHANNEL}" \
  --channel "${PYLIBCUDF_CHANNEL}" \
  --channel "${PYLIBRAFT_CHANNEL}" \
  "libcugraph=${RAPIDS_VERSION_MAJOR_MINOR}.*" \
  "libcugraph_etl=${RAPIDS_VERSION_MAJOR_MINOR}.*" \
  "libcugraph-tests=${RAPIDS_VERSION_MAJOR_MINOR}.*"

rapids-logger "Check GPU usage"
nvidia-smi

# RAPIDS_DATASET_ROOT_DIR is used by test scripts
export RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
pushd "${RAPIDS_DATASET_ROOT_DIR}"
./get_test_data.sh --cpp_ci_subset
popd

export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/

# Skip benchmark tests inside of CI
export GTEST_FILTER="-*benchmark*"

# Run libcugraph gtests from libcugraph-tests package
rapids-logger "Run gtests"
./ci/run_ctests.sh -j10 && EXITCODE=$? || EXITCODE=$?;

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
