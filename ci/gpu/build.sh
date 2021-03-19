#!/usr/bin/env bash
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
    $WORKSPACE/build.sh -v clean libcugraph cugraph
else
    export LIBCUGRAPH_BUILD_DIR="$WORKSPACE/ci/artifacts/cugraph/cpu/conda_work/cpp/build"

    # Faiss patch
    echo "Update libcugraph.so"
    cd $LIBCUGRAPH_BUILD_DIR
    chrpath -d libcugraph.so
    patchelf --replace-needed `patchelf --print-needed libcugraph.so | grep faiss` libfaiss.so libcugraph.so

    CONDA_FILE=`find $WORKSPACE/ci/artifacts/cugraph/cpu/conda-bld/ -name "libcugraph*.tar.bz2"`
    CONDA_FILE=`basename "$CONDA_FILE" .tar.bz2` #get filename without extension
    CONDA_FILE=${CONDA_FILE//-/=} #convert to conda install
    echo "Installing $CONDA_FILE"
    conda install -c $WORKSPACE/ci/artifacts/cugraph/cpu/conda-bld/ "$CONDA_FILE"

    echo "Build cugraph..."
    $WORKSPACE/build.sh cugraph
fi

################################################################################
# TEST - Run GoogleTest and py.tests for libcugraph and cuGraph
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
