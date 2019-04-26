#!/usr/bin/env bash
# Copyright (c) 2018, NVIDIA CORPORATION.
##########################################
# cuGraph GPU build & testscript for CI  #
##########################################
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path, build parallel level, and CUDA version
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDA_VERSION_SHORT=${CUDA_VERSION%.*}
export CUDF_VERSION=0.6

# Set home to the job's workspace
export HOME=$WORKSPACE

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
source activate gdf
conda install -c nvidia/label/cuda$CUDA_VERSION_SHORT -c rapidsai/label/cuda$CUDA_VERSION_SHORT -c rapidsai-nightly/label/cuda$CUDA_VERSION_SHORT -c numba -c conda-forge -c defaults cudf=$CUDF_VERSION nvgraph networkx python-louvain

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# BUILD - Build libcugraph and cuGraph from source
################################################################################

logger "Build libcugraph..."
mkdir -p $WORKSPACE/cpp/build
cd $WORKSPACE/cpp/build
logger "Run cmake libcugraph..."
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CXX11_ABI=ON ..

logger "Clean up make..."
make clean

logger "Make libcugraph..."
make -j${PARALLEL_LEVEL}

logger "Install libcugraph..."
make -j${PARALLEL_LEVEL} install

logger "Build cuGraph..."
cd $WORKSPACE/python
python setup.py install

################################################################################
# TEST - Run GoogleTest and py.tests for libcugraph and cuGraph
################################################################################

logger "Check GPU usage..."
nvidia-smi

logger "GoogleTest for libcugraph..."
cd $WORKSPACE/cpp/build
GTEST_OUTPUT="xml:${WORKSPACE}/test-results/" gtests/GDFGRAPH_TEST

logger "Python py.test for cuGraph..."
cd $WORKSPACE/python
py.test --cache-clear --junitxml=${WORKSPACE}/junit-cugraph.xml -v
