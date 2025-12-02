#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

conda config --file /opt/conda/.condarc --remove-key custom_multichannels
conda config --get
cat /opt/conda/.condarc

# Support invoking test_python.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-github python)

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
  --numprocesses=8 \
  --dist=worksteal \
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
rapids-logger "pytest cugraph (not mg, with xdist)"
./ci/run_cugraph_pytests.sh \
  --verbose \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cugraph.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  -m "not mg" \
  -k "not test_bulk_sampler and not test_create_undirected_graph_from_asymmetric_adj_list and not test_node2vec and not(test_property_graph.py and cyber)" \
  --cov-config=../../.coveragerc \
  --cov=cugraph \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cugraph-coverage.xml" \
  --cov-report=term

# excludes known failures that will always fail when run in combination
rapids-logger "pytest cugraph (mg, with xdist)"
./ci/run_cugraph_pytests.sh \
  --verbose \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cugraph.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  -m "mg" \
  -k "not test_property_graph_mg and not test_dist_sampler_mg" \
  --cov-config=../../.coveragerc \
  --cov=cugraph \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cugraph-coverage.xml" \
  --cov-report=term

rapids-logger "pytest cugraph (mg dist_sampler and uns)"
./ci/run_cugraph_pytests.sh \
  --verbose \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cugraph.xml" \
  -m "mg" \
  -k "test_dist_sampler_mg" \
  --cov-config=../../.coveragerc \
  --cov=cugraph \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cugraph-coverage.xml" \
  --cov-report=term

rapids-logger "pytest cugraph benchmarks (run as tests)"
./ci/run_cugraph_benchmark_pytests.sh --verbose

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
