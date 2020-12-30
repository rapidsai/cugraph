#!/usr/bin/env bash
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
##########################################
# cuGraph GPU build & testscript for CI  #
##########################################
set -e
set -o pipefail
NUMARGS=$#
ARGS=$*

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path, build parallel level, and CUDA version
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
export CUDA_REL=${CUDA_VERSION%.*}

function cleanup {
  gpuci_logger "Removing datasets and temp files"
  rm -rf $WORKSPACE/datasets/test
  rm -rf $WORKSPACE/datasets/benchmark
  rm -f testoutput.txt
}

# Set cleanup trap for Jenkins
if [ ! -z "$JENKINS_HOME" ] ; then
  gpuci_logger "Jenkins environment detected, setting cleanup trap"
  trap cleanup EXIT
fi

# Set home to the job's workspace
export HOME=$WORKSPACE

# Parse git describe
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Install dependencies"
gpuci_conda_retry install -y \
      "libcudf=${MINOR_VERSION}" \
      "cudf=${MINOR_VERSION}" \
      "librmm=${MINOR_VERSION}" \
      "rmm=${MINOR_VERSION}" \
      "cudatoolkit=$CUDA_REL" \
      "dask-cudf=${MINOR_VERSION}" \
      "dask-cuda=${MINOR_VERSION}" \
      "ucx-py=${MINOR_VERSION}" \
      "ucx-proc=*=gpu" \
      "rapids-build-env=$MINOR_VERSION.*" \
      "rapids-notebook-env=$MINOR_VERSION.*" \
      rapids-pytest-benchmark

# https://docs.rapids.ai/maintainers/depmgmt/
# gpuci_conda_retry remove --force rapids-build-env rapids-notebook-env
# gpuci_conda_retry install -y "your-pkg=1.0.0"

gpuci_logger "Install the master version of dask and distributed"
pip install "git+https://github.com/dask/distributed.git" --upgrade --no-deps
pip install "git+https://github.com/dask/dask.git" --upgrade --no-deps

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

################################################################################
# BUILD - Build libcugraph and cuGraph from source
################################################################################

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
  gpuci_logger "Build from source"
  $WORKSPACE/build.sh -v clean libcugraph cugraph --allgpuarch
fi

################################################################################
# TEST - Run GoogleTest and py.tests for libcugraph and cuGraph
################################################################################

set +e -Eo pipefail
EXITCODE=0
trap "EXITCODE=1" ERR

if hasArg --skip-tests; then
    gpuci_logger "Skipping Tests"
else
    gpuci_logger "Check GPU usage"
    nvidia-smi

    # If this is a PR build, skip downloading large datasets and don't run the
    # slow-running tests that use them.
    # See: https://docs.rapids.ai/maintainers/gpuci/#environment-variables
    if [ "$BUILD_MODE" = "pull-request" ]; then
        TEST_MODE_FLAG="--quick"
    else
        TEST_MODE_FLAG=""
    fi

    ${WORKSPACE}/ci/test.sh ${TEST_MODE_FLAG} | tee testoutput.txt

    echo -e "\nTOP 20 SLOWEST TESTS:\n"
    # Wrap in echo to prevent non-zero exit since this command is non-essential
    echo "$(${WORKSPACE}/ci/getGTestTimes.sh testoutput.txt | head -20)"

    ${WORKSPACE}/ci/gpu/test-notebooks.sh 2>&1 | tee nbtest.log
    python ${WORKSPACE}/ci/utils/nbtestlog2junitxml.py nbtest.log
fi

if [ -n "\${CODECOV_TOKEN}" ]; then
    codecov -t \$CODECOV_TOKEN
fi

return ${EXITCODE}
