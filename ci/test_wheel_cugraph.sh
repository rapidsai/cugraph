#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eoxu pipefail

source rapids-init-pip

# Download the cugraph package built in the previous step
CUGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cugraph cugraph --stable --cuda "$RAPIDS_CUDA_VERSION")")

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

if [[ "${LIBCUGRAPH_FROM_NIGHTLY:-false}" == "true" || "${PYLIBCUGRAPH_FROM_NIGHTLY:-false}" == "true" ]]; then
  rapids-generate-version > ./VERSION
  RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
fi

if [[ "${LIBCUGRAPH_FROM_NIGHTLY:-false}" == "true" ]]; then
  # libcugraph wasn't rebuilt for this PR; resolve it from the nightly wheel index instead.
  LIBCUGRAPH_SPEC=("libcugraph-${RAPIDS_PY_CUDA_SUFFIX}==${RAPIDS_PACKAGE_VERSION}.*")
else
  LIBCUGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcugraph cugraph --cuda "$RAPIDS_CUDA_VERSION")")
  LIBCUGRAPH_SPEC=("${LIBCUGRAPH_WHEELHOUSE}"/libcugraph*.whl)
fi

if [[ "${PYLIBCUGRAPH_FROM_NIGHTLY:-false}" == "true" ]]; then
  # pylibcugraph wasn't rebuilt for this PR; resolve it from the nightly wheel index instead.
  PYLIBCUGRAPH_SPEC=("pylibcugraph-${RAPIDS_PY_CUDA_SUFFIX}==${RAPIDS_PACKAGE_VERSION}.*")
else
  PYLIBCUGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python pylibcugraph cugraph --stable --cuda "$RAPIDS_CUDA_VERSION")")
  PYLIBCUGRAPH_SPEC=("${PYLIBCUGRAPH_WHEELHOUSE}"/pylibcugraph*.whl)
fi

if [[ "${LIBCUGRAPH_FROM_NIGHTLY:-false}" == "true" || "${PYLIBCUGRAPH_FROM_NIGHTLY:-false}" == "true" ]]; then
  PIP_INSTALL_ARGS+=("--extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple")
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
    "${PYLIBCUGRAPH_SPEC[@]}" \
    "${LIBCUGRAPH_SPEC[@]}" \
    "${PIP_INSTALL_ARGS[@]}"

./ci/test_wheel.sh cugraph
