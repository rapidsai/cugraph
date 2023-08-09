#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-print-env

rapids-mamba-retry install \
    --channel "${CPP_CHANNEL}" \
    libcugraph libcugraph_etl libcugraph-tests

rapids-logger "Check GPU usage"
nvidia-smi

# RAPIDS_DATASET_ROOT_DIR is used by test scripts
export RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
pushd "${RAPIDS_DATASET_ROOT_DIR}"
./get_test_data.sh --subset
popd

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/

# Run libcugraph gtests from libcugraph-tests package
rapids-logger "Run gtests"
cd "$CONDA_PREFIX"/bin/gtests/libcugraph/
ctest -j10 --output-on-failure

if [ -d "$CONDA_PREFIX"/bin/gtests/libcugraph_c/ ]; then
  cd "$CONDA_PREFIX"/bin/gtests/libcugraph_c/
  ctest -j10 --output-on-failure
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
