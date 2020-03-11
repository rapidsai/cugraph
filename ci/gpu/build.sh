#!/usr/bin/env bash
# Copyright (c) 2018, NVIDIA CORPORATION.
##########################################
# cuGraph GPU build & testscript for CI  #
##########################################
set -e
set -o pipefail
NUMARGS=$#
ARGS=$*

function logger {
  echo -e "\n>>>> $@\n"
}

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function cleanup {
  logger "Removing datasets and temp files..."
  rm -rf $WORKSPACE/datasets/test
  rm -rf $WORKSPACE/datasets/benchmark
  rm -f testoutput.txt
}

# Set cleanup trap for Jenkins
if [ ! -z "$JENKINS_HOME" ] ; then
  logger "Jenkins environment detected, setting cleanup trap..."
  trap cleanup EXIT
fi

# Set path, build parallel level, and CUDA version
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDA_REL=${CUDA_VERSION%.*}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Parse git describe
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
source activate gdf

logger "conda install required packages"
conda install -c nvidia -c rapidsai -c rapidsai-nightly -c conda-forge -c defaults \
      cudf=${MINOR_VERSION} \
      rmm=${MINOR_VERSION} \
      networkx>=2.3 \
      python-louvain \
      fa2l \
      cudatoolkit=$CUDA_REL \
      dask>=2.12.0 \
      distributed>=2.12.0 \
      dask-cudf=${MINOR_VERSION} \
      dask-cuda=${MINOR_VERSION} \
      nccl>=2.5 \
      libcypher-parser \
      ipython=7.3* \
      jupyterlab

# Install the master version of dask and distributed
logger "pip install git+https://github.com/dask/distributed.git --upgrade --no-deps"
pip install "git+https://github.com/dask/distributed.git" --upgrade --no-deps

logger "pip install git+https://github.com/dask/dask.git --upgrade --no-deps"
pip install "git+https://github.com/dask/dask.git" --upgrade --no-deps


logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# BUILD - Build libcugraph and cuGraph from source
################################################################################

logger "Build libcugraph..."
$WORKSPACE/build.sh clean libcugraph cugraph

################################################################################
# TEST - Run GoogleTest and py.tests for libcugraph and cuGraph
################################################################################

if hasArg --skip-tests; then
    logger "Skipping Tests..."
else
    logger "Check GPU usage..."
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
