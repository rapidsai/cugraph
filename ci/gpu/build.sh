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

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4

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

export CUDA_VERSION_SHORT=${CUDA_VERSION%.*}
conda install -c nvidia -c rapidsai -c rapidsai-nightly/label/cuda$CUDA_VERSION_SHORT -c numba -c conda-forge -c defaults cudf=0.6 nvgraph

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
cd $WORKSPACE
tar -zxvf cpp/src/tests/datasets.tar.gz -C /
py.test --cache-clear --junitxml=${WORKSPACE}/junit-cugraph.xml -v
