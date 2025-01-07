#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -eoxu pipefail

# Download the packages built in the previous step
mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist
RAPIDS_PY_WHEEL_NAME="libcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./local-libcugraph-dep

# TODO(jameslamb): remove this when https://github.com/rapidsai/raft/pull/2531 is merged
source ./ci/use_wheels_from_prs.sh

python -m pip install \
    "$(echo ./dist/pylibcugraph*.whl)[test]" \
    ./local-libcugraph-dep/libcugraph*.whl

./ci/test_wheel.sh pylibcugraph
