#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eoxu pipefail

source rapids-init-pip

# Download the packages built in the previous step
PYLIBCUGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python pylibcugraph cugraph --stable --cuda "$RAPIDS_CUDA_VERSION")")
LIBCUGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcugraph cugraph --cuda "$RAPIDS_CUDA_VERSION")")

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

python -m venv libcugraph-env
. libcugraph-env/bin/activate

rapids-pip-retry install \
    -v \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "${LIBCUGRAPH_WHEELHOUSE}"/libcugraph*.whl
python -c "import libcugraph; libcugraph.load_library()"
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
    "${LIBCUGRAPH_WHEELHOUSE}"/libcugraph*.whl

./ci/test_wheel.sh pylibcugraph
