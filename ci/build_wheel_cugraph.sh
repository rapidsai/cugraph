#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

RAPIDS_PY_WHEEL_NAME=pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX} rapids-download-wheels-from-s3 ./local-pylibcugraph
python -m pip install --no-deps ./local-pylibcugraph/pylibcugraph*.whl

export SKBUILD_CONFIGURE_OPTIONS="-DDETECT_CONDA_ENV=OFF -DCUGRAPH_BUILD_WHEELS=ON -DFIND_CUGRAPH_CPP=OFF -DCPM_cugraph-ops_SOURCE=${GITHUB_WORKSPACE}/cugraph-ops/"

./ci/build_wheel.sh cugraph python/cugraph
