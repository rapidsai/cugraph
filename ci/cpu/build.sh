#!/bin/bash
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
#########################################
# cuGraph CPU conda build script for CI #
#########################################
set -e

# Set path and build parallel level
# openmpi dir is required on CentOS for finding MPI libs from cmake
if [[ -e /etc/os-release ]] && (grep -qi centos /etc/os-release); then
    export PATH=/opt/conda/bin:/usr/local/cuda/bin:/usr/lib64/openmpi/bin:$PATH
else
    export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
fi
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# If nightly build, append current YYMMDD to version
if [[ "$BUILD_MODE" = "branch" && "$SOURCE_BRANCH" = branch-* ]] ; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

# Setup 'gpuci_conda_retry' for build retries (results in 2 total attempts)
export GPUCI_CONDA_RETRY_MAX=1
export GPUCI_CONDA_RETRY_SLEEP=30

# Use Ninja to build
export CMAKE_GENERATOR="Ninja"
export CONDA_BLD_DIR="${WORKSPACE}/.conda-bld"

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment variables"
env

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Remove rapidsai-nightly channel if we are building main branch
if [ "$SOURCE_BRANCH" = "main" ]; then
  conda config --system --remove channels rapidsai-nightly
fi

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

###############################################################################
# BUILD - Conda package builds
###############################################################################

gpuci_logger "Build conda package for libcugraph and libcugraph_etl"
if [ "$BUILD_LIBCUGRAPH" == '1' ]; then
  if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    gpuci_conda_retry build --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/libcugraph
    gpuci_conda_retry build --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/libcugraph_etl
  else
    gpuci_conda_retry build --no-build-id --croot ${CONDA_BLD_DIR} --dirty --no-remove-work-dir conda/recipes/libcugraph
    gpuci_conda_retry build --no-build-id --croot ${CONDA_BLD_DIR} --dirty --no-remove-work-dir conda/recipes/libcugraph_etl
    mkdir -p ${CONDA_BLD_DIR}/libcugraph/work
    cp -r ${CONDA_BLD_DIR}/work/* ${CONDA_BLD_DIR}/libcugraph/work
  fi
fi

gpuci_logger "Build conda packages for pylibcugraph and cugraph"
if [ "$BUILD_CUGRAPH" == "1" ]; then
  if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    gpuci_conda_retry build --croot ${CONDA_BLD_DIR} conda/recipes/pylibcugraph --python=$PYTHON
    gpuci_conda_retry build --croot ${CONDA_BLD_DIR} conda/recipes/cugraph --python=$PYTHON
  else
    gpuci_conda_retry build --croot ${CONDA_BLD_DIR} conda/recipes/pylibcugraph -c ci/artifacts/cugraph/cpu/.conda-bld/ --dirty --no-remove-work-dir --python=$PYTHON
    gpuci_conda_retry build --croot ${CONDA_BLD_DIR} conda/recipes/cugraph -c ci/artifacts/cugraph/cpu/.conda-bld/ --dirty --no-remove-work-dir --python=$PYTHON
  fi
fi

################################################################################
# UPLOAD - Conda packages
################################################################################

gpuci_logger "Upload conda packages"
source ci/cpu/upload.sh
