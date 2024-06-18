#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

PARALLEL_LEVEL=$(python -c \
  "from math import ceil; from multiprocessing import cpu_count; print(ceil(cpu_count()/2))")

export SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF;-DFIND_CUGRAPH_CPP=OFF;-DCPM_cugraph-ops_SOURCE=${GITHUB_WORKSPACE}/cugraph-ops/"
export SKBUILD_BUILD_TOOL_ARGS="-j{PARALLEL_LEVEL};-l{PARALLEL_LEVEL}"

./ci/build_wheel.sh pylibcugraph python/pylibcugraph
