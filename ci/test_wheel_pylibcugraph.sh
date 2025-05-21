#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -eoxu pipefail

source ci/use_gha_tools_from_pr.sh

# Download the packages built in the previous step
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
LIBCUGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBCUGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
    "$(echo "${PYLIBCUGRAPH_WHEELHOUSE}"/pylibcugraph*.whl)[test]" \
    "${LIBCUGRAPH_WHEELHOUSE}"/libcugraph*.whl

./ci/test_wheel.sh pylibcugraph
