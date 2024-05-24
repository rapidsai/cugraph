#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

version=$(rapids-generate-version)

rapids-logger "Begin cpp build"

PARALLEL_LEVEL="${PARALLEL_LEVEL:-$(nproc --all --ignore=1)}" \
RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild conda/recipes/libcugraph

rapids-upload-conda-to-s3 cpp
