#!/bin/bash
set -e

# Check environment
source ci/check_environment.sh


# GPU Test Stage
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

gpuci_mamba_retry install \
  -c "${CPP_CHANNEL}" \
  -c "${PYTHON_CHANNEL}" \
  cugraph libcugraph libcugraph-tests

TESTRESULTS_DIR=test-results
mkdir -p ${TESTRESULTS_DIR}
SUITEERROR=0

gpuci_logger "Check GPU usage"
nvidia-smi

set +e
gpuci_logger "Running googletests"
# run gtests from libcugraph-tests package
for gt in "$CONDA_PREFIX/bin/gtests/libcugraph/"* ; do
    ${gt} --gtest_output=xml:${TESTRESULTS_DIR}/
    exitcode=$?
    if (( ${exitcode} != 0 )); then
        SUITEERROR=${exitcode}
        echo "FAILED: GTest ${gt}"
    fi
done

cd python

gpuci_logger "pytest cugraph"
py.test --cache-clear --junitxml=test-results/junit-cugraph.xml -v --cov-config=.coveragerc --cov=cugraph --cov-report=xml:python/cugraph-coverage.xml --cov-report term
exitcode=$?
if (( ${exitcode} != 0 )); then
    SUITEERROR=${exitcode}
    echo "FAILED: 1 or more tests in /cugraph/python"
fi

exit ${SUITEERROR}
