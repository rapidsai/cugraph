#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

export SKBUILD_CONFIGURE_OPTIONS="-DDETECT_CONDA_ENV=OFF -DCUGRAPH_BUILD_WHEELS=ON -DFIND_CUGRAPH_CPP=OFF -DCPM_cugraph-ops_SOURCE=${GITHUB_WORKSPACE}/cugraph-ops/"

./ci/build_wheel.sh pylibcugraph python/pylibcugraph
