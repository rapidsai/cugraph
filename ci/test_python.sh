#!/bin/bash

set -euo pipefail

#TODO: Remove
. /opt/conda/etc/profile.d/conda.sh
conda activate base

# Check environment
source ci/check_env.sh

# GPU Test Stage
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

gpuci_mamba_retry install \
  -c "${CPP_CHANNEL}" \
  -c "${PYTHON_CHANNEL}" \
  cugraph pylibcugraph

# Install test dependencies
gpuci_mamba_retry install pytest pytest-cov

gpuci_logger "Check GPU usage"
nvidia-smi

set +e
set -E
trap "EXITCODE=1" ERR

gpuci_logger "Running cuGraph python tests"
"${GITHUB_WORKSPACE}/ci/test.sh" --quick --run-python-tests | tee testoutput.txt

gpuci_logger "Running cuGraph notebook test script..."
"${GITHUB_WORKSPACE}/ci/gpu/test-notebooks.sh" 2>&1 | tee nbtest.log

python "${GITHUB_WORKSPACE}/ci/utils/nbtestlog2junitxml.py" nbtest.log

exit "${EXITCODE}"
