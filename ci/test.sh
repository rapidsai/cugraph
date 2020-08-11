#!/bin/bash

# note: do not use set -e in order to allow all gtest invocations to take place,
# and instead keep track of exit status and exit with an overall exit status
set -o pipefail

NUMARGS=$#
ARGS=$*
THISDIR=$(cd $(dirname $0);pwd)
CUGRAPH_ROOT=$(cd ${THISDIR}/..;pwd)
GTEST_ARGS="--gtest_output=xml:${CUGRAPH_ROOT}/test-results/"
DOWNLOAD_MODE=""
ERRORCODE=0

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
    ERRORCODE=$((ERRORCODE | $?))
    # no need to run tests if dataset download fails
    if (( ${ERRORCODE} != 0 )); then
        exit ${ERRORCODE}
    fi
fi

cd ${CUGRAPH_ROOT}/cpp/build

for gt in gtests/*; do
    test_name=$(basename $gt)
    echo "Running GoogleTest $test_name"
    ${gt} ${GTEST_FILTER} ${GTEST_ARGS}
    ERRORCODE=$((ERRORCODE | $?))
done

echo "Python pytest for cuGraph..."
cd ${CUGRAPH_ROOT}/python
pytest --cache-clear --junitxml=${CUGRAPH_ROOT}/junit-cugraph.xml -v --cov-config=.coveragerc --cov=cugraph --cov-report=xml:${WORKSPACE}/python/cugraph/cugraph-coverage.xml --cov-report term --ignore=cugraph/raft
ERRORCODE=$((ERRORCODE | $?))

echo "Python benchmarks for cuGraph (running as tests)..."
cd ${CUGRAPH_ROOT}/benchmarks
pytest -v -m "managedmem_on and poolallocator_on and tiny" --benchmark-disable
ERRORCODE=$((ERRORCODE | $?))

exit ${ERRORCODE}
