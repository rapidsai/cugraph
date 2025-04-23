#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking test_python.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

export LD_PRELOAD="${CONDA_PREFIX}/lib/libgomp.so.1"

# RAPIDS_DATASET_ROOT_DIR is used by test scripts
RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
export RAPIDS_DATASET_ROOT_DIR

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest pylibcugraph"
./ci/run_pylibcugraph_pytests.sh \
  --verbose \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-pylibcugraph.xml" \
  --cov-config=../../.coveragerc \
  --cov=pylibcugraph \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/pylibcugraph-coverage.xml" \
  --cov-report=term


# Test runs that include tests that use dask require
# --import-mode=append. Those tests start a LocalCUDACluster that inherits
# changes from pytest's modifications to PYTHONPATH (which defaults to
# prepending source tree paths to PYTHONPATH).  This causes the
# LocalCUDACluster subprocess to import cugraph from the source tree instead of
# the install location, and in most cases, the source tree does not have
# extensions built in-place and will result in ImportErrors.
#
# FIXME: TEMPORARILY disable MG PropertyGraph tests (experimental) tests and
# bulk sampler IO tests (hangs in CI)
rapids-logger "pytest cugraph"
./ci/run_cugraph_pytests.sh \
  --verbose \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cugraph.xml" \
  --cov-config=../../.coveragerc \
  --cov=cugraph \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cugraph-coverage.xml" \
  --cov-report=term


rapids-logger "pytest cugraph benchmarks (run as tests)"
./ci/run_cugraph_benchmark_pytests.sh --verbose

rapids-logger "pytest cugraph-service (single GPU)"
./ci/run_cugraph_service_pytests.sh \
  --verbose \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cugraph-service.xml" \
  --cov-config=../.coveragerc \
  --cov=cugraph_service_client \
  --cov=cugraph_service_server \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cugraph-service-coverage.xml" \
  --cov-report=term

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
