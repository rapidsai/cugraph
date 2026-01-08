#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eoxu pipefail

source rapids-init-pip

# Download the packages built in the previous step
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
  PYLIBCUGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" pylibcugraph --stable --cuda "$RAPIDS_CUDA_VERSION")")
  CUGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" cugraph --stable --cuda "$RAPIDS_CUDA_VERSION")")
  source ./ci/use_upstream_sabi_wheels.sh
else
  PYLIBCUGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
  CUGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
fi

LIBCUGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
    "$(echo "${CUGRAPH_WHEELHOUSE}"/cugraph*.whl)[test]" \
    "${PYLIBCUGRAPH_WHEELHOUSE}"/pylibcugraph*.whl \
    "${LIBCUGRAPH_WHEELHOUSE}"/libcugraph*.whl

./ci/test_wheel.sh cugraph
