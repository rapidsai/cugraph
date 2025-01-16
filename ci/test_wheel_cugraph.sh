#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -eoxu pipefail

# Download the packages built in the previous step
mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist
RAPIDS_PY_WHEEL_NAME="pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./local-pylibcugraph-dep

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install \
    "$(echo ./dist/cugraph*.whl)[test]" \
    ./local-pylibcugraph-dep/pylibcugraph*.whl

./ci/test_wheel.sh cugraph
