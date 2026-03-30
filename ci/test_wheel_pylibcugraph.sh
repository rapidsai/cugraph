#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eoxu pipefail

source rapids-init-pip

# Download the packages built in the previous step
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
PYLIBCUGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" pylibcugraph --stable --cuda "$RAPIDS_CUDA_VERSION")")
LIBCUGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${PYLIBCUGRAPH_WHEELHOUSE}"/pylibcugraph*.whl)[test]" \
    "${LIBCUGRAPH_WHEELHOUSE}"/libcugraph*.whl

./ci/test_wheel.sh pylibcugraph
