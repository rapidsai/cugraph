#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

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
  nx-cugraph \
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

# FIXME: TEMPORARILY disable single-GPU "MG" testing
rapids-logger "pytest cugraph"
pushd python/cugraph/cugraph
DASK_WORKER_DEVICES="0" \
DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL="1000s" \
DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="1000s" \
DASK_CUDA_WAIT_WORKERS_MIN_TIMEOUT="1000s" \
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

rapids-logger "pytest nx-cugraph"
pushd python/nx-cugraph/nx_cugraph
pytest \
  --capture=no \
  --verbose \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-nx-cugraph.xml" \
  --cov-config=../../.coveragerc \
  --cov=nx_cugraph \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/nx-cugraph-coverage.xml" \
  --cov-report=term \
  --benchmark-disable \
  tests
popd

rapids-logger "pytest networkx using nx-cugraph backend"
pushd python/nx-cugraph
./run_nx_tests.sh
# run_nx_tests.sh outputs coverage data, so check that total coverage is >0.0%
# in case nx-cugraph failed to load but fallback mode allowed the run to pass.
_coverage=$(coverage report|grep "^TOTAL")
echo "nx-cugraph coverage from networkx tests: $_coverage"
echo $_coverage | awk '{ if ($NF == "0.0%") exit 1 }'
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

    # Will automatically install built dependencies of cuGraph-PyG
    rapids-mamba-retry install \
      --channel "${CPP_CHANNEL}" \
      --channel "${PYTHON_CHANNEL}" \
      --channel pytorch \
      --channel nvidia \
      --channel pyg \
      --channel rapidsai-nightly \
      "cugraph-pyg" \
      "pytorch>=2.0,<2.1" \
      "pytorch-cuda=11.8"

    # Install pyg dependencies (which requires pip)
    pip install \
        pyg_lib \
        torch_scatter \
        torch_sparse \
        torch_cluster \
        torch_spline_conv \
      -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

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
