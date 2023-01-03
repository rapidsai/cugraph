#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

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
SUITEERROR=0

rapids-print-env

rapids-mamba-retry install \
    --channel "${CPP_CHANNEL}" \
    libcugraph libcugraph_etl libcugraph-tests

rapids-logger "Check GPU usage"
nvidia-smi

# RAPIDS_DATASET_ROOT_DIR is used by test scripts
export RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
pushd "${RAPIDS_DATASET_ROOT_DIR}"
./get_test_data.sh
popd

# Run libcugraph gtests from libcugraph-tests package
rapids-logger "Run gtests"
set +e

# TODO: exit code handling is too verbose. Find a cleaner solution.

for gt in "$CONDA_PREFIX"/bin/gtests/libcugraph/* ; do
    test_name=$(basename ${gt})
    echo "Running gtest $test_name"
    ${gt} --gtest_output=xml:${RAPIDS_TESTS_DIR}

    exitcode=$?
    if (( ${exitcode} != 0 )); then
        SUITEERROR=${exitcode}
        echo "FAILED: GTest ${gt}"
    fi
done

for ct in "$CONDA_PREFIX"/bin/gtests/libcugraph_c/CAPI_*_TEST ; do
    test_name=$(basename ${ct})
    echo "Running C API test $test_name"
    ${ct}

    exitcode=$?
    if (( ${exitcode} != 0 )); then
        SUITEERROR=${exitcode}
        echo "FAILED: C API test ${ct}"
    fi
done

exit ${SUITEERROR}
