#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking test_cpp.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

. /opt/conda/etc/profile.d/conda.sh

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" \
  --prepend-channel "${CPP_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

# RAPIDS_DATASET_ROOT_DIR is used by test scripts
RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
export RAPIDS_DATASET_ROOT_DIR
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
exit "${EXITCODE}"
