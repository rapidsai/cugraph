#!/usr/bin/env bash
# Copyright (c) 2018, NVIDIA CORPORATION.
##########################################
# cuGraph GPU build & testscript for CI  #
##########################################
set -e
NUMARGS=$#
ARGS=$*

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

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
conda install -y -c nvidia -c rapidsai -c rapidsai-nightly -c numba -c conda-forge \
      cudf=${MINOR_VERSION} \
      rmm=${MINOR_VERSION} \
      networkx>=2.3 \
      python-louvain \
      cudatoolkit=$CUDA_REL \
      dask>=2.1.0 \
      distributed>=2.1.0 \
      dask-cudf=${MINOR_VERSION} \
      dask-cuda=${MINOR_VERSION} \
      libcypher-parser

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
# CACHE - Load build directories from previous PR, reducing build time
################################################################################

if [ -f $WORKSPACE/cache.tgz ] ; then
    cd $WORKSPACE
    logger "Cache detected, extracting..."
    rm -rf cpp/build python/build
    tar xzvfp cache.tgz
    rm cache.tgz
    cp cpp/build/Makefile /tmp/cache-time-ref
    
    # Patch CMake files for current paths
    cd cpp/build
    grep -rlZ -E "${JENKINS_HOME}/.*/cpp" . | xargs -0 sed -i "s|${JENKINS_HOME}/.*/cpp|${HOME}/cpp|g"
    
    # Touch all build files to bring the timestamps back to normal after sed
    grep -rlZ -E "${HOME}/cpp" . | xargs -0 touch -r /tmp/cache-time-ref
    ls -la cpp/build
    
    # Return to workspace
    cd $WORKSPACE
fi

################################################################################
# BUILD - Build libcugraph and cuGraph from source
################################################################################

logger "Build libcugraph and cugraph..."
$WORKSPACE/build.sh libcugraph cugraph

################################################################################
# TEST - Run GoogleTest and py.tests for libcugraph and cuGraph
################################################################################

if hasArg --skip-tests; then
    logger "Skipping Tests..."
else
    logger "Check GPU usage..."
    nvidia-smi

    logger "GoogleTest for libcugraph..."
    cd $WORKSPACE/cpp/build
    GTEST_OUTPUT="xml:${WORKSPACE}/test-results/" gtests/GDFGRAPH_TEST

    logger "Python py.test for cuGraph..."
    cd $WORKSPACE/python
    py.test --cache-clear --junitxml=${WORKSPACE}/junit-cugraph.xml -v
fi

################################################################################
# CACHE - Upload build results for T4 testing
################################################################################

logger "Creating cache..."
cd $WORKSPACE
tar czvfp cache.tgz cpp/build python/build
