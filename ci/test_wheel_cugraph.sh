#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eoxu pipefail

source rapids-init-pip

# Download the packages built in the previous step
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

PYLIBCUGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" pylibcugraph --stable --cuda "$RAPIDS_CUDA_VERSION")")
CUGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" cugraph --stable --cuda "$RAPIDS_CUDA_VERSION")")
LIBCUGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

# Update this when 'torch' publishes CUDA wheels supporting newer CTKs.
#
# See notes in 'dependencies.yaml' for details on supported versions.
CUDA_MAJOR=$(echo "${RAPIDS_CUDA_VERSION}" | cut -d'.' -f1)
CUDA_MINOR=$(echo "${RAPIDS_CUDA_VERSION}" | cut -d'.' -f2)
PIP_INSTALL_ARGS=()
if \
    { [ "${CUDA_MAJOR}" -eq 12 ] && [ "${CUDA_MINOR}" -eq 9 ]; } \
    || { [ "${CUDA_MAJOR}" -eq 13 ] && [ "${CUDA_MINOR}" -eq 0 ]; }; \
then
    # ensure a CUDA variant of 'torch' is used
    rapids-logger "Downloading PyTorch CUDA wheels"
    TORCH_WHEEL_DIR="$(mktemp -d)"
    ./ci/download-torch-wheels.sh "${TORCH_WHEEL_DIR}"
    PIP_INSTALL_ARGS+=("${TORCH_WHEEL_DIR}"/torch*.whl)
fi

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${CUGRAPH_WHEELHOUSE}"/cugraph*.whl)[test]" \
    "${PYLIBCUGRAPH_WHEELHOUSE}"/pylibcugraph*.whl \
    "${LIBCUGRAPH_WHEELHOUSE}"/libcugraph*.whl \
    "${PIP_INSTALL_ARGS[@]}"

./ci/test_wheel.sh cugraph
