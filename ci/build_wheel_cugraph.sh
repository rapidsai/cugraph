#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

package_dir="python/cugraph"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# env variable 'PIP_CONSTRAINT' is set up by rapids-init-pip. It constrains all subsequent
# 'pip install', 'pip download', etc. calls (except the isolated wheel build,
# which is handled separately by ci/build_wheel.sh)
#
# A skipped 'wheel-build-libcugraph'/'wheel-build-pylibcugraph' job (that package
# unaffected by this PR) means no artifact was uploaded this run. Fail fast (short
# retry budget) rather than waiting out the full retry window, then fall back to
# resolving that package from the nightly wheel index instead.
NEEDS_NIGHTLY_INDEX=false

if LIBCUGRAPH_WHEELHOUSE=$(RAPIDS_RETRY_MAX=1 RAPIDS_RETRY_SLEEP=15 rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcugraph cugraph --cuda "$RAPIDS_CUDA_VERSION")" 2>/tmp/libcugraph_wheel_download.log); then
  cat >> "${PIP_CONSTRAINT}" <<EOF
libcugraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUGRAPH_WHEELHOUSE}"/libcugraph_*.whl)
EOF
else
  cat /tmp/libcugraph_wheel_download.log >&2
  rapids-logger "No libcugraph wheel found for this run; resolving it from the nightly wheel index"
  rapids-generate-version > ./VERSION
  RAPIDS_PACKAGE_MINOR_VERSION=$(cut -d. -f1,2 ./VERSION)
  cat >> "${PIP_CONSTRAINT}" <<EOF
libcugraph-${RAPIDS_PY_CUDA_SUFFIX}==${RAPIDS_PACKAGE_MINOR_VERSION}.*,>=0.0.0a0
EOF
  NEEDS_NIGHTLY_INDEX=true
fi

if PYLIBCUGRAPH_WHEELHOUSE=$(RAPIDS_RETRY_MAX=1 RAPIDS_RETRY_SLEEP=15 rapids-download-from-github "$(rapids-artifact-name wheel_python pylibcugraph cugraph --stable --cuda "$RAPIDS_CUDA_VERSION")" 2>/tmp/pylibcugraph_wheel_download.log); then
  cat >> "${PIP_CONSTRAINT}" <<EOF
pylibcugraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${PYLIBCUGRAPH_WHEELHOUSE}"/pylibcugraph_*.whl)
EOF
else
  cat /tmp/pylibcugraph_wheel_download.log >&2
  rapids-logger "No pylibcugraph wheel found for this run; resolving it from the nightly wheel index"
  if [[ -z "${RAPIDS_PACKAGE_MINOR_VERSION:-}" ]]; then
    rapids-generate-version > ./VERSION
    RAPIDS_PACKAGE_MINOR_VERSION=$(cut -d. -f1,2 ./VERSION)
  fi
  cat >> "${PIP_CONSTRAINT}" <<EOF
pylibcugraph-${RAPIDS_PY_CUDA_SUFFIX}==${RAPIDS_PACKAGE_MINOR_VERSION}.*,>=0.0.0a0
EOF
  NEEDS_NIGHTLY_INDEX=true
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
