#!/bin/bash
# Copyright (c) 2019-2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Any failing command will set EXITCODE to non-zero
set -e           # abort the script on error, this will change for running tests (see below)
set -o pipefail  # piped commands propagate their error
set -E           # ERR traps are inherited by subcommands
trap "EXITCODE=1" ERR

NUMARGS=$#
ARGS=$*
THISDIR=$(cd $(dirname $0);pwd)
CUGRAPH_ROOT=$(cd ${THISDIR}/..;pwd)
GTEST_ARGS="--gtest_output=xml:${CUGRAPH_ROOT}/test-results/"
DOWNLOAD_MODE=""
EXITCODE=0

export RAPIDS_DATASET_ROOT_DIR=${CUGRAPH_ROOT}/datasets

# FIXME: consider using getopts for option parsing
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Add options unique to running a "quick" subset of tests here:
#  - pass --subset flag to download script to skip large downloads
#  - filter the "huge" dataset tests
if hasArg "--quick"; then
    echo "Running \"quick\" tests only..."
    DOWNLOAD_MODE="--subset"
    GTEST_FILTER="--gtest_filter=-hibench_test/Tests_MGSpmv_hibench.CheckFP32_hibench*:*huge*"
else
    echo "Running all tests..."
    # FIXME: do we still need to always filter these tests?
    GTEST_FILTER="--gtest_filter=-hibench_test/Tests_MGSpmv_hibench.CheckFP32_hibench*"
fi

if hasArg "--skip-download"; then
    echo "Using datasets in ${RAPIDS_DATASET_ROOT_DIR}"
else
    echo "Download datasets..."
    cd ${RAPIDS_DATASET_ROOT_DIR}
    bash ./get_test_data.sh ${DOWNLOAD_MODE}
fi

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    cd ${CUGRAPH_ROOT}/cpp/build
else
    export LD_LIBRARY_PATH="$WORKSPACE/ci/artifacts/cugraph/cpu/conda_work/cpp/build:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    cd $WORKSPACE/ci/artifacts/cugraph/cpu/conda_work/cpp/build
fi

# FIXME: if possible, any install and build steps should be moved outside this
# script since a failing install/build step is treated as a failing test command
# and will not stop the script. This script is also only expected to run tests
# in a preconfigured environment, and install/build steps are unexpected side
# effects.
if [[ "$PROJECT_FLASH" == "1" ]]; then
    export LIBCUGRAPH_BUILD_DIR="$WORKSPACE/ci/artifacts/cugraph/cpu/conda_work/cpp/build"

    # Faiss patch
    echo "Update libcugraph.so"
    cd $LIBCUGRAPH_BUILD_DIR
    chrpath -d libcugraph.so
    chrpath -d libcugraph++.so
    patchelf --replace-needed `patchelf --print-needed libcugraph++.so | grep faiss` libfaiss.so libcugraph++.so

    echo "Update test binaries"
    for gt in tests/*_TEST; do
      chrpath -d ${gt}
      patchelf --replace-needed `patchelf --print-needed ${gt} | grep faiss` libfaiss.so ${gt}
    done

    CONDA_FILE=`find $WORKSPACE/ci/artifacts/cugraph/cpu/conda-bld/ -name "libcugraph*.tar.bz2"`
    CONDA_FILE=`basename "$CONDA_FILE" .tar.bz2` #get filename without extension
    CONDA_FILE=${CONDA_FILE//-/=} #convert to conda install
    echo "Installing $CONDA_FILE"
    conda install -c $WORKSPACE/ci/artifacts/cugraph/cpu/conda-bld/ "$CONDA_FILE"

    echo "Build cugraph..."
    $WORKSPACE/build.sh cugraph
fi

# Do not abort the script on error from this point on. This allows all tests to
# run regardless of pass/fail, but relies on the ERR trap above to manage the
# EXITCODE for the script.
set +e

echo "C++ gtests for cuGraph..."
for gt in tests/*_TEST; do
    test_name=$(basename $gt)
    echo "Running gtest $test_name"
    ${gt} ${GTEST_FILTER} ${GTEST_ARGS}
    echo "Ran gtest $test_name : return code was: $?, test script exit code is now: $EXITCODE"
done

echo "Python pytest for cuGraph..."
cd ${CUGRAPH_ROOT}/python
pytest --cache-clear --junitxml=${CUGRAPH_ROOT}/junit-cugraph.xml -v --cov-config=.coveragerc --cov=cugraph --cov-report=xml:${WORKSPACE}/python/cugraph/cugraph-coverage.xml --cov-report term --ignore=cugraph/raft --benchmark-disable
echo "Ran Python pytest for cugraph : return code was: $?, test script exit code is now: $EXITCODE"

echo "Python benchmarks for cuGraph (running as tests)..."
cd ${CUGRAPH_ROOT}/benchmarks
pytest -v -m "managedmem_on and poolallocator_on and tiny" --benchmark-disable
echo "Ran Python benchmarks for cuGraph (running as tests) : return code was: $?, test script exit code is now: $EXITCODE"

echo "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
