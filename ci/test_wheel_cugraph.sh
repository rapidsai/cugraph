#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eoxu pipefail

source rapids-init-pip

# Download the cugraph package built in the previous step
CUGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cugraph cugraph --stable --cuda "$RAPIDS_CUDA_VERSION")")

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

# A skipped 'wheel-build-libcugraph'/'wheel-build-pylibcugraph' job (that package
# unaffected by this PR) means no artifact was uploaded this run. Fail fast (short
# retry budget) rather than waiting out the full retry window, then fall back to
# resolving that package from the nightly wheel index instead.
NEEDS_NIGHTLY_INDEX=false

if LIBCUGRAPH_WHEELHOUSE=$(RAPIDS_RETRY_MAX=1 RAPIDS_RETRY_SLEEP=15 rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcugraph cugraph --cuda "$RAPIDS_CUDA_VERSION")" 2>/tmp/libcugraph_wheel_download.log); then
  LIBCUGRAPH_SPEC=("${LIBCUGRAPH_WHEELHOUSE}"/libcugraph*.whl)
else
  cat /tmp/libcugraph_wheel_download.log >&2
  rapids-logger "No libcugraph wheel found for this run; resolving it from the nightly wheel index"
  rapids-generate-version > ./VERSION
  RAPIDS_PACKAGE_MINOR_VERSION=$(cut -d. -f1,2 ./VERSION)
  LIBCUGRAPH_SPEC=("libcugraph-${RAPIDS_PY_CUDA_SUFFIX}==${RAPIDS_PACKAGE_MINOR_VERSION}.*,>=0.0.0a0")
  NEEDS_NIGHTLY_INDEX=true
fi

if PYLIBCUGRAPH_WHEELHOUSE=$(RAPIDS_RETRY_MAX=1 RAPIDS_RETRY_SLEEP=15 rapids-download-from-github "$(rapids-artifact-name wheel_python pylibcugraph cugraph --stable --cuda "$RAPIDS_CUDA_VERSION")" 2>/tmp/pylibcugraph_wheel_download.log); then
  PYLIBCUGRAPH_SPEC=("${PYLIBCUGRAPH_WHEELHOUSE}"/pylibcugraph*.whl)
else
  cat /tmp/pylibcugraph_wheel_download.log >&2
  rapids-logger "No pylibcugraph wheel found for this run; resolving it from the nightly wheel index"
  if [[ -z "${RAPIDS_PACKAGE_MINOR_VERSION:-}" ]]; then
    rapids-generate-version > ./VERSION
    RAPIDS_PACKAGE_MINOR_VERSION=$(cut -d. -f1,2 ./VERSION)
  fi
  PYLIBCUGRAPH_SPEC=("pylibcugraph-${RAPIDS_PY_CUDA_SUFFIX}==${RAPIDS_PACKAGE_MINOR_VERSION}.*,>=0.0.0a0")
  NEEDS_NIGHTLY_INDEX=true
fi

if [[ "${NEEDS_NIGHTLY_INDEX}" == "true" ]]; then
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
