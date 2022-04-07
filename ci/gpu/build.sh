#!/usr/bin/env bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
##########################################
# cuGraph GPU build & testscript for CI  #
##########################################
set -e           # abort the script on error, this will change for running tests (see below)
set -o pipefail  # piped commands propagate their error
NUMARGS=$#
ARGS=$*

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path, build parallel level, and CUDA version
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
export CUDA_REL=${CUDA_VERSION%.*}
export CONDA_ARTIFACT_PATH=${WORKSPACE}/ci/artifacts/cugraph/cpu/.conda-bld/

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

# ucx-py version
export UCX_PY_VERSION='0.26.*'

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
export PATH=$(conda info --base)/envs/rapids/bin:$PATH

gpuci_logger "Install dependencies"
# Assume libcudf and librmm will be installed via cudf and rmm respectively.
# This is done to prevent the following install scenario:
# libcudf = 22.04.00a220315, cudf = 22.04.00a220308
# where cudf 220308 was chosen possibly because it has fewer/different
# dependencies and the corresponding recipes have specified these combinations
# should work when sometimes they do not.
# FIXME: remove testing label when gpuCI has the ability to move the pyraft
# label from testing to main.
gpuci_mamba_retry install -c rapidsai-nightly/label/testing -y \
      "cudf=${MINOR_VERSION}" \
      "rmm=${MINOR_VERSION}" \
      "libraft-headers=${MINOR_VERSION}" \
      "pyraft=${MINOR_VERSION}" \
      "cudatoolkit=$CUDA_REL" \
      "dask-cudf=${MINOR_VERSION}" \
      "dask-cuda=${MINOR_VERSION}" \
      "ucx-py=${UCX_PY_VERSION}" \
      "ucx-proc=*=gpu" \
      "rapids-build-env=$MINOR_VERSION.*" \
      "rapids-notebook-env=$MINOR_VERSION.*" \
      rapids-pytest-benchmark

# https://docs.rapids.ai/maintainers/depmgmt/
# gpuci_mamba_retry remove --force rapids-build-env rapids-notebook-env
# gpuci_mamba_retry install -y "your-pkg=1.0.0"

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

################################################################################
# BUILD
################################################################################

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    gpuci_logger "Build from source"
    $WORKSPACE/build.sh -v clean libcugraph pylibcugraph cugraph
else
    echo "Installing libcugraph-tests"
    gpuci_mamba_retry install -c ${CONDA_ARTIFACT_PATH} libcugraph libcugraph_etl libcugraph-tests

    gpuci_logger "Install the master version of dask and distributed"
    pip install "git+https://github.com/dask/distributed.git" --upgrade --no-deps
    pip install "git+https://github.com/dask/dask.git" --upgrade --no-deps

    echo "Build pylibcugraph and cugraph..."
    $WORKSPACE/build.sh pylibcugraph cugraph
fi

################################################################################
# TEST
################################################################################

# Switch to +e to allow failing commands to continue the script, which is needed
# so all testing commands run regardless of pass/fail. This requires the desired
# exit code to be managed using the ERR trap.
set +e           # allow script to continue on error
set -E           # ERR traps are inherited by subcommands
trap "EXITCODE=1" ERR

EXITCODE=0

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

    gpuci_logger "Running cuGraph test.sh..."
    ${WORKSPACE}/ci/test.sh ${TEST_MODE_FLAG} | tee testoutput.txt
    gpuci_logger "Ran cuGraph test.sh : return code was: $?, gpu/build.sh exit code is now: $EXITCODE"

    gpuci_logger "Running cuGraph notebook test script..."
    ${WORKSPACE}/ci/gpu/test-notebooks.sh 2>&1 | tee nbtest.log
    gpuci_logger "Ran cuGraph notebook test script : return code was: $?, gpu/build.sh exit code is now: $EXITCODE"
    python ${WORKSPACE}/ci/utils/nbtestlog2junitxml.py nbtest.log
fi

if [ -n "${CODECOV_TOKEN}" ]; then
    codecov -t $CODECOV_TOKEN
fi

gpuci_logger "gpu/build.sh returning value: $EXITCODE"
return ${EXITCODE}
