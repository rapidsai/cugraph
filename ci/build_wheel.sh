#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_name=$1
package_dir=$2

source rapids-configure-sccache
source rapids-date-string
source rapids-init-pip

export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="${package_name}/${RAPIDS_CONDA_ARCH}/cuda${RAPIDS_CUDA_VERSION%%.*}/wheel/preprocessor-cache"
export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true

rapids-generate-version > ./VERSION

cd "${package_dir}"

sccache --stop-server 2>/dev/null || true

rapids-logger "Building '${package_name}' wheel"

rapids-pip-retry wheel \
    -w dist \
    -v \
    --no-deps \
    --disable-pip-version-check \
    --extra-index-url https://pypi.nvidia.com \
    .

sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

EXCLUDE_ARGS=(
  --exclude "libraft.so"
)

# Avoid picking up dependencies on CUDA wheels that come through
# transitively from 'libraft'.
#
# 'libraft' wheels are responsible for carrying a runtime dependency on
# these based on RAFT's needs.
EXCLUDE_ARGS+=(
  --exclude "libcublas.so.*"
  --exclude "libcublasLt.so.*"
  --exclude "libcurand.so.*"
  --exclude "libcusolver.so.*"
  --exclude "libcusparse.so.*"
  --exclude "libnvJitLink.so.*"
  --exclude "librapids_logger.so"
  --exclude "librmm.so"
)

if [[ "${package_dir}" != "python/libcugraph" ]]; then
    EXCLUDE_ARGS+=(
      --exclude "libcugraph_c.so"
      --exclude "libcugraph.so"
    )
fi

# repair wheels and write to the location that artifact-uploading code expects to find them
python -m auditwheel repair -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" "${EXCLUDE_ARGS[@]}" dist/*
