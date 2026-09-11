#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

package_dir="python/cugraph"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# env variable 'PIP_CONSTRAINT' is set up by rapids-init-pip. It constrains all subsequent
# 'pip install', 'pip download', etc. calls (except those used in 'pip wheel', handled separately in build scripts)
NEEDS_NIGHTLY_INDEX=false

if [[ "${LIBCUGRAPH_FROM_NIGHTLY:-false}" == "true" || "${PYLIBCUGRAPH_FROM_NIGHTLY:-false}" == "true" ]]; then
  rapids-generate-version > ./VERSION
  RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
fi

if [[ "${LIBCUGRAPH_FROM_NIGHTLY:-false}" == "true" ]]; then
  # libcugraph wasn't rebuilt for this PR; resolve it from the nightly wheel index instead.
  cat >> "${PIP_CONSTRAINT}" <<EOF
libcugraph-${RAPIDS_PY_CUDA_SUFFIX}==${RAPIDS_PACKAGE_VERSION}.*
EOF
  NEEDS_NIGHTLY_INDEX=true
else
  # Download the libcugraph wheel built in the previous step and make it
  # available for pip to find.
  LIBCUGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcugraph cugraph --cuda "$RAPIDS_CUDA_VERSION")")
  cat >> "${PIP_CONSTRAINT}" <<EOF
libcugraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUGRAPH_WHEELHOUSE}"/libcugraph_*.whl)
EOF
fi

if [[ "${PYLIBCUGRAPH_FROM_NIGHTLY:-false}" == "true" ]]; then
  # pylibcugraph wasn't rebuilt for this PR; resolve it from the nightly wheel index instead.
  cat >> "${PIP_CONSTRAINT}" <<EOF
pylibcugraph-${RAPIDS_PY_CUDA_SUFFIX}==${RAPIDS_PACKAGE_VERSION}.*
EOF
  NEEDS_NIGHTLY_INDEX=true
else
  # Download the pylibcugraph wheel built in the previous step and make it
  # available for pip to find.
  PYLIBCUGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python pylibcugraph cugraph --stable --cuda "$RAPIDS_CUDA_VERSION")")
  cat >> "${PIP_CONSTRAINT}" <<EOF
pylibcugraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${PYLIBCUGRAPH_WHEELHOUSE}"/pylibcugraph_*.whl)
EOF
fi

if [[ "${NEEDS_NIGHTLY_INDEX}" == "true" ]]; then
  cat >> "${PIP_CONSTRAINT}" <<EOF
--extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple
EOF
fi

# TODO: move this variable into `ci-wheel`
# Format Python limited API version string
RAPIDS_PY_API="cp${RAPIDS_PY_VERSION//./}"
export RAPIDS_PY_API

./ci/build_wheel.sh cugraph ${package_dir} --stable
./ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_python cugraph cugraph --stable --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
