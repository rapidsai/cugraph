#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eoxu pipefail

source rapids-init-pip

# Download the pylibcugraph package built in the previous step
PYLIBCUGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python pylibcugraph cugraph --stable --cuda "$RAPIDS_CUDA_VERSION")")

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

# A skipped 'wheel-build-libcugraph' job (libcugraph unaffected by this PR) means no
# artifact was uploaded this run. Fail fast (short retry budget) rather than waiting
# out the full retry window, then fall back to resolving libcugraph from the nightly
# wheel index instead.
if LIBCUGRAPH_WHEELHOUSE=$(RAPIDS_RETRY_MAX=1 RAPIDS_RETRY_SLEEP=15 rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcugraph cugraph --cuda "$RAPIDS_CUDA_VERSION")" 2>/tmp/libcugraph_wheel_download.log); then
  LIBCUGRAPH_SPEC=("${LIBCUGRAPH_WHEELHOUSE}"/libcugraph*.whl)
else
  cat /tmp/libcugraph_wheel_download.log >&2
  rapids-logger "No libcugraph wheel found for this run; resolving it from the nightly wheel index"
  rapids-generate-version > ./VERSION
  RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
  LIBCUGRAPH_SPEC=("libcugraph-${RAPIDS_PY_CUDA_SUFFIX}==${RAPIDS_PACKAGE_VERSION}.*" "--extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple")
fi

python -m venv libcugraph-env
. libcugraph-env/bin/activate

rapids-pip-retry install \
    -v \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "${LIBCUGRAPH_SPEC[@]}"
python -c "import libcugraph; assert (libraries := libcugraph.load_library()) and all(libraries)"
deactivate

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
    "${LIBCUGRAPH_SPEC[@]}"

./ci/test_wheel.sh pylibcugraph
