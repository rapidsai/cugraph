#!/bin/bash
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
#########################################
# cuGraph CPU conda build script for CI #
#########################################
set -e

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
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

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment variables"
env

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

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

gpuci_logger "Build conda pkg for libcugraph"
if [ "$BUILD_LIBCUGRAPH" == '1' ]; then
  if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    conda build conda/recipes/libcugraph
  else
    conda build --dirty --no-remove-work-dir conda/recipes/libcugraph
  fi
fi

gpuci_logger "Build conda pkg for cugraph"
if [ "$BUILD_CUGRAPH" == "1" ]; then
  if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    conda build conda/recipes/cugraph --python=$PYTHON
  else
    conda build conda/recipes/cugraph -c ci/artifacts/cugraph/cpu/conda-bld/ --dirty --no-remove-work-dir --python=$PYTHON
  fi
fi

################################################################################
# UPLOAD - Conda packages
################################################################################

gpuci_logger "Upload conda packages"
source ci/cpu/upload.sh
