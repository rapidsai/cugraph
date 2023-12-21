#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

export SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF;-DFIND_CUGRAPH_CPP=OFF;-DCPM_cugraph-ops_SOURCE=${GITHUB_WORKSPACE}/cugraph-ops/"

./ci/build_wheel.sh pylibcugraph python/pylibcugraph
