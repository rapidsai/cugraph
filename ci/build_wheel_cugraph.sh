#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Download the pylibcugraph wheel built in the previous step and make it
# available for pip to find. We must use PIP_FIND_LINKS because the package
# must be made available to the isolated build step, and there is no way to
# manually install it into that environment.
RAPIDS_PY_WHEEL_NAME=pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX} rapids-download-wheels-from-s3 ./local-pylibcugraph
export PIP_FIND_LINKS=$(pwd)/local-pylibcugraph

export SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF;-DFIND_CUGRAPH_CPP=OFF;-DCPM_cugraph-ops_SOURCE=${GITHUB_WORKSPACE}/cugraph-ops/"

./ci/build_wheel.sh cugraph python/cugraph
