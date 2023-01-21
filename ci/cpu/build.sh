#!/bin/bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

# Workaround to keep Jenkins builds working
# until we migrate fully to GitHub Actions
export RAPIDS_CUDA_VERSION="${CUDA}"
export SCCACHE_BUCKET=rapids-sccache
export SCCACHE_REGION=us-west-2
export SCCACHE_IDLE_TIMEOUT=32768

# Use Ninja to build
export CMAKE_GENERATOR="Ninja"
export CONDA_BLD_DIR="${WORKSPACE}/.conda-bld"

# ucx-py version
export UCX_PY_VERSION='0.30.*'

# Whether to keep `dask/label/dev` channel in the env. If INSTALL_DASK_MAIN=0,
# `dask/label/dev` channel is removed.
export INSTALL_DASK_MAIN=1

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
  conda config --system --remove channels dask/label/dev
elif [[ "${INSTALL_DASK_MAIN}" == 0 ]]; then
  # Remove `dask/label/dev` channel if INSTALL_DASK_MAIN=0
  conda config --system --remove channels dask/label/dev
fi

gpuci_logger "Check versions"
python --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

# FIXME: for now, force the building of all packages so they are built on a
# machine with a single CUDA version, then have the gpu/build.sh script simply
# install. This should eliminate a mismatch between different CUDA versions on
# cpu vs. gpu builds that is problematic with CUDA 11.5 Enhanced Compat.
if [ "$BUILD_LIBCUGRAPH" == '1' ]; then
  BUILD_CUGRAPH=1
  # If we are doing CUDA + Python builds, libcugraph package is located at ${CONDA_BLD_DIR}
  CONDA_LOCAL_CHANNEL="${CONDA_BLD_DIR}"
else
  # If we are doing Python builds only, libcugraph package is placed here by Project Flash
  CONDA_LOCAL_CHANNEL="ci/artifacts/cugraph/cpu/.conda-bld/"
fi

# FIXME: Move boa install to gpuci/rapidsai
gpuci_mamba_retry install -c conda-forge boa

###############################################################################
# BUILD - Conda package builds
###############################################################################

if [ "$BUILD_LIBCUGRAPH" == '1' ]; then
  gpuci_logger "Building conda package for libcugraph and libcugraph_etl"
  if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/libcugraph
  else
    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} --dirty --no-remove-work-dir conda/recipes/libcugraph
    mkdir -p ${CONDA_BLD_DIR}/libcugraph
    mv ${CONDA_BLD_DIR}/work ${CONDA_BLD_DIR}/libcugraph/work
  fi
  gpuci_logger "sccache stats"
  sccache --show-stats
else
  gpuci_logger "SKIPPING build of conda package for libcugraph and libcugraph_etl"
fi

if [ "$BUILD_CUGRAPH" == "1" ]; then
  gpuci_logger "Building conda packages for pylibcugraph, cugraph, cugraph-service, and cugraph-dgl"
  if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    gpuci_logger "pylibcugraph"
    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/pylibcugraph --python=$PYTHON
    gpuci_logger "cugraph"
    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/cugraph --python=$PYTHON
    gpuci_logger "cugraph-service"
    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/cugraph-service --python=$PYTHON
    gpuci_logger "cugraph-pyg"
    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/cugraph-pyg --python=$PYTHON
    gpuci_logger "cugraph-dgl"
    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/cugraph-dgl --python=$PYTHON -c dglteam -c pytorch
  else
    gpuci_logger "pylibcugraph"
    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/pylibcugraph -c ${CONDA_LOCAL_CHANNEL} --dirty --no-remove-work-dir --python=$PYTHON
    gpuci_logger "cugraph"
    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/cugraph -c ${CONDA_LOCAL_CHANNEL} --dirty --no-remove-work-dir --python=$PYTHON
    gpuci_logger "cugraph-service"
    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/cugraph-service -c ${CONDA_LOCAL_CHANNEL} --dirty --no-remove-work-dir --python=$PYTHON
    gpuci_logger "cugraph-pyg"
    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/cugraph-pyg -c ${CONDA_LOCAL_CHANNEL} --dirty --no-remove-work-dir --python=$PYTHON
    gpuci_logger "cugraph-dgl"
    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/cugraph-dgl -c ${CONDA_LOCAL_CHANNEL} --dirty --no-remove-work-dir --python=$PYTHON -c dglteam -c pytorch

    mkdir -p ${CONDA_BLD_DIR}/cugraph
    mv ${CONDA_BLD_DIR}/work ${CONDA_BLD_DIR}/cugraph/work
  fi
else
  gpuci_logger "SKIPPING build of conda packages for pylibcugraph, cugraph, cugraph-service, and cugraph-dgl"
fi

################################################################################
# UPLOAD - Conda packages
################################################################################

# Uploads disabled due to new GH Actions implementation
# gpuci_logger "Upload conda pkgs"
# source ci/cpu/upload.sh
