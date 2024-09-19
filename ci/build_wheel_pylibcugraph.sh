#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

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

source ./ci/use_wheels_from_prs.sh

export SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF;-DFIND_CUGRAPH_CPP=OFF;-DCPM_cugraph-ops_SOURCE=${GITHUB_WORKSPACE}/cugraph-ops/${EXTRA_CMAKE_ARGS}"
export SKBUILD_BUILD_TOOL_ARGS="-j${PARALLEL_LEVEL};-l${PARALLEL_LEVEL}"

./ci/build_wheel.sh pylibcugraph python/pylibcugraph
