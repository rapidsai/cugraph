#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-print-env

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  libcugraph \
  pylibcugraph \
  cugraph \
  cugraph-service-server \
  cugraph-service-client

rapids-logger "Check GPU usage"
nvidia-smi

# RAPIDS_DATASET_ROOT_DIR is used by test scripts
export RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
pushd "${RAPIDS_DATASET_ROOT_DIR}"
./get_test_data.sh --benchmark
popd

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest pylibcugraph"
pushd python/pylibcugraph/pylibcugraph
pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-pylibcugraph.xml" \
  --cov-config=../../.coveragerc \
  --cov=pylibcugraph \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/pylibcugraph-coverage.xml" \
  --cov-report=term \
  tests
popd

rapids-logger "pytest cugraph"
pushd python/cugraph/cugraph
export DASK_WORKER_DEVICES="0"
pytest \
  -v \
  --benchmark-disable \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cugraph.xml" \
  --cov-config=../../.coveragerc \
  --cov=cugraph \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cugraph-coverage.xml" \
  --cov-report=term \
  -k "not test_property_graph_mg" \
  tests
popd

rapids-logger "pytest cugraph benchmarks (run as tests)"
pushd benchmarks
pytest \
  --capture=no \
  --verbose \
  -m tiny \
  --benchmark-disable \
  cugraph/pytest-based/bench_algos.py
popd

rapids-logger "pytest cugraph-service (single GPU)"
pushd python/cugraph-service
pytest \
  --capture=no \
  --verbose \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cugraph-service.xml" \
  --cov-config=../.coveragerc \
  --cov=cugraph_service_client \
  --cov=cugraph_service_server \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cugraph-service-coverage.xml" \
  --cov-report=term \
  --benchmark-disable \
  -k "not mg" \
  tests
popd

if [[ "${RAPIDS_CUDA_VERSION}" == "11.8.0" ]]; then
  if [[ "${RUNNER_ARCH}" != "ARM64" ]]; then
    # we are only testing in a single cuda version
    # because of pytorch and rapids compatibilty problems
    rapids-mamba-retry env create --force -f env.yaml -n test_cugraph_dgl

    # activate test_cugraph_dgl environment for dgl
    set +u
    conda activate test_cugraph_dgl
    set -u
    rapids-mamba-retry install \
      --channel "${CPP_CHANNEL}" \
      --channel "${PYTHON_CHANNEL}" \
      --channel pytorch \
      --channel pytorch-nightly \
      --channel dglteam/label/cu118 \
      --channel nvidia \
      libcugraph \
      pylibcugraph \
      pylibcugraphops \
      cugraph \
      cugraph-dgl \
      'dgl>=1.1.0.cu*' \
      'pytorch>=2.0' \
      'pytorch-cuda>=11.8'

    rapids-print-env

    rapids-logger "pytest cugraph_dgl (single GPU)"
    pushd python/cugraph-dgl/tests
    pytest \
      --cache-clear \
      --ignore=mg \
      --junitxml="${RAPIDS_TESTS_DIR}/junit-cugraph-dgl.xml" \
      --cov-config=../../.coveragerc \
      --cov=cugraph_dgl \
      --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cugraph-dgl-coverage.xml" \
      --cov-report=term \
      .
    popd

    # Reactivate the test environment back
    set +u
    conda deactivate
    conda activate test
    set -u
  else
    rapids-logger "skipping cugraph_dgl pytest on ARM64"
  fi
else
  rapids-logger "skipping cugraph_dgl pytest on CUDA!=11.8"
fi

if [[ "${RAPIDS_CUDA_VERSION}" == "11.8.0" ]]; then
  if [[ "${RUNNER_ARCH}" != "ARM64" ]]; then
    rapids-mamba-retry env create --force -f env.yaml -n test_cugraph_pyg

    # Temporarily allow unbound variables for conda activation.
    set +u
    conda activate test_cugraph_pyg
    set -u

    # Install pytorch
    rapids-mamba-retry install \
      --force-reinstall \
      --channel pyg \
      --channel pytorch \
      --channel nvidia \
      'pyg=2.3' \
      'pytorch>=2.0' \
      'pytorch-cuda>=11.8'

    rapids-mamba-retry install \
      --channel "${CPP_CHANNEL}" \
      --channel "${PYTHON_CHANNEL}" \
      libcugraph \
      pylibcugraph \
      pylibcugraphops \
      cugraph \
      cugraph-pyg

    rapids-print-env

    rapids-logger "pytest cugraph_pyg (single GPU)"
    pushd python/cugraph-pyg/cugraph_pyg
    # rmat is not tested because of multi-GPU testing
    pytest \
      --cache-clear \
      --ignore=tests/int \
      --ignore=tests/mg \
      --junitxml="${RAPIDS_TESTS_DIR}/junit-cugraph-pyg.xml" \
      --cov-config=../../.coveragerc \
      --cov=cugraph_pyg \
      --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cugraph-pyg-coverage.xml" \
      --cov-report=term \
      .
    popd

    # Reactivate the test environment back
    set +u
    conda deactivate
    conda activate test
    set -u

  else
    rapids-logger "skipping cugraph_pyg pytest on ARM64"
  fi
else
  rapids-logger "skipping cugraph_pyg pytest on CUDA != 11.8"
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
