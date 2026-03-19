#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# [description]
#
#   Downloads a CUDA variant of 'torch' from the correct index, based on CUDA major version.
#
#   This exists to avoid using 'pip --extra-index-url', which could allow for CPU-only 'torch'
#   to be downloaded from pypi.org.
#

set -e -u -o pipefail

TORCH_WHEEL_DIR="${1}"

# Ensure CUDA-enabled 'torch' packages are always used.
#
# Downloading + passing the downloaded file as a requirement forces the use of this
# package, so we don't accidentally end up with a CPU-only 'torch' from 'pypi.org'
# (which can happen because pip doesn't support index priority).
#
# Not appending this to PIP_CONSTRAINT, because we don't want the torch '--extra-index-url'
# to leak outside of this script into other 'pip {download,install}' calls.
rapids-dependency-file-generator \
    --output requirements \
    --file-key "torch_only" \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES};require_gpu=true" \
| tee ./torch-constraints.txt

rapids-pip-retry download \
  --isolated \
  --prefer-binary \
  --no-deps \
  -d "${TORCH_WHEEL_DIR}" \
  --constraint "${PIP_CONSTRAINT}" \
  --constraint ./torch-constraints.txt \
  'torch'
