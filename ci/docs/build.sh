#!/bin/bash
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
#################################
# cuGraph Docs build script for CI #
#################################

if [ -z "$PROJECT_WORKSPACE" ]; then
    echo ">>>> ERROR: Could not detect PROJECT_WORKSPACE in environment"
    echo ">>>> WARNING: This script contains git commands meant for automated building, do not run locally"
    exit 1
fi

export DOCS_WORKSPACE=$WORKSPACE/docs
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export HOME=$WORKSPACE
export PROJECT_WORKSPACE=/rapids/cugraph
export LIBCUDF_KERNEL_CACHE_PATH="$HOME/.jitify-cache"
export PROJECTS=(cugraph libcugraph)

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Show conda info"
conda info
conda config --show-sources
conda list --show-channel-urls

# Build Doxygen docs
gpuci_logger "Build Doxygen docs"
wget "https://raw.githubusercontent.com/rapidsai/docs/gh-pages/api/librmm/${BRANCH_VERSION}/rmm.tag" || echo "Failed to download rmm Doxygen tag"
cd $PROJECT_WORKSPACE/cpp/build
cmake --build . -t docs_cugraph

# Build Python docs
gpuci_logger "Build Sphinx docs"
cd $PROJECT_WORKSPACE/docs/cugraph
make html

# Commit to Website
cd $DOCS_WORKSPACE

for PROJECT in ${PROJECTS[@]}; do
    if [ ! -d "api/$PROJECT/$BRANCH_VERSION" ]; then
        mkdir -p api/$PROJECT/$BRANCH_VERSION
    fi
    rm -rf $DOCS_WORKSPACE/api/$PROJECT/$BRANCH_VERSION/*
done


mv $PROJECT_WORKSPACE/cpp/doxygen/html/* $DOCS_WORKSPACE/api/libcugraph/$BRANCH_VERSION
mv $PROJECT_WORKSPACE/docs/cugraph/build/html/* $DOCS_WORKSPACE/api/cugraph/$BRANCH_VERSION

