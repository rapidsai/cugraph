#!/bin/bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh

set -e

# Setup 'gpuci_retry' for upload retries (results in 4 total attempts)
export GPUCI_RETRY_MAX=3
export GPUCI_RETRY_SLEEP=30

# Set default label options if they are not defined elsewhere
export LABEL_OPTION=${LABEL_OPTION:-"--label main"}

# Skip uploads unless BUILD_MODE == "branch"
if [ ${BUILD_MODE} != "branch" ]; then
  echo "Skipping upload"
  return 0
fi

# Skip uploads if there is no upload key
if [ -z "$MY_UPLOAD_KEY" ]; then
  echo "No upload key"
  return 0
fi

################################################################################
# UPLOAD - Conda packages
################################################################################

gpuci_logger "Starting conda uploads"

if [[ "$BUILD_LIBCUGRAPH" == "1" && "$UPLOAD_LIBCUGRAPH" == "1" ]]; then
  LIBCUGRAPH_FILES=$(conda build --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/libcugraph --output)
  echo "Upload libcugraph"
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing --no-progress ${LIBCUGRAPH_FILES}
fi

if [[ "$BUILD_CUGRAPH" == "1" ]]; then
  PYLIBCUGRAPH_FILE=$(conda build --croot ${CONDA_BLD_DIR} conda/recipes/pylibcugraph --python=$PYTHON --output)
  test -e ${PYLIBCUGRAPH_FILE}
  echo "Upload pylibcugraph file: ${PYLIBCUGRAPH_FILE}"
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${PYLIBCUGRAPH_FILE} --no-progress
 
  CUGRAPH_FILE=$(conda build --croot ${CONDA_BLD_DIR} conda/recipes/cugraph --python=$PYTHON --output)
  test -e ${CUGRAPH_FILE}
  echo "Upload cugraph file: ${CUGRAPH_FILE}"
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${CUGRAPH_FILE} --no-progress
 
  CUGRAPH_SERVICE_FILES=$(conda build --croot ${CONDA_BLD_DIR} conda/recipes/cugraph-service --python=$PYTHON --output)
  # no test -e since CUGRAPH_SERVICE_FILES has multiple files
  echo "Upload cugraph-server files: ${CUGRAPH_SERVICE_FILES}"
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing --no-progress ${CUGRAPH_SERVICE_FILES}
  
  CUGRAPH_PYG_FILE=$(conda build --croot ${CONDA_BLD_DIR} conda/recipes/cugraph-pyg --python=$PYTHON --output)
  test -e ${CUGRAPH_PYG_FILE}
  echo "Upload cugraph file: ${CUGRAPH_PYG_FILE}"
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${CUGRAPH_PYG_FILE} --no-progress

  CUGRAPH_DGL_FILE=$(conda build --croot ${CONDA_BLD_DIR} conda/recipes/cugraph-dgl --python=$PYTHON --output)
  test -e ${CUGRAPH_DGL_FILE}
  echo "Upload cugraph-dgl file: ${CUGRAPH_DGL_FILE}"
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${CUGRAPH_DGL_FILE} --no-progress
fi
