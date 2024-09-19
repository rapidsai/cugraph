#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

source ./ci/use_wheels_from_prs.sh

# Download the pylibcugraph wheel built in the previous step and make it
# available for pip to find.
#
# ensure 'cugraph' wheel builds always use the 'pylibcugraph' just built in the same CI run
#
# using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment
CPP_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 /tmp/pylibcugraph_dist)

echo "pylibcugraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${CPP_WHEELHOUSE}/pylibcugraph_*.whl)" >> ./constraints.txt
export PIP_CONSTRAINT="${PWD}/constraints.txt"

PARALLEL_LEVEL=$(python -c \
  "from math import ceil; from multiprocessing import cpu_count; print(ceil(cpu_count()/4))")
case "${RAPIDS_CUDA_VERSION}" in
  12.*)
    EXTRA_CMAKE_ARGS=";-DUSE_CUDA_MATH_WHEELS=ON"
  ;;
  11.*)
    EXTRA_CMAKE_ARGS=";-DUSE_CUDA_MATH_WHEELS=OFF"
  ;;
esac

export SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF;-DFIND_CUGRAPH_CPP=OFF;-DCPM_cugraph-ops_SOURCE=${GITHUB_WORKSPACE}/cugraph-ops/${EXTRA_CMAKE_ARGS}"
export SKBUILD_BUILD_TOOL_ARGS="-j${PARALLEL_LEVEL};-l${PARALLEL_LEVEL}"

./ci/build_wheel.sh cugraph python/cugraph
