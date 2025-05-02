#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2
package_type=$3

source rapids-configure-sccache
source rapids-date-string

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

rapids-generate-version > ./VERSION

cd "${package_dir}"

sccache --zero-stats

rapids-logger "Building '${package_name}' wheel"

rapids-pip-retry wheel \
    -w dist \
    -v \
    --no-deps \
    --disable-pip-version-check \
    --extra-index-url https://pypi.nvidia.com \
    .

sccache --show-adv-stats

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
)

if [[ "${package_dir}" != "python/libcugraph" ]]; then
    EXCLUDE_ARGS+=(
      --exclude "libcugraph_c.so"
      --exclude "libcugraph.so"
    )
fi

# repair wheels and write to the location that artifact-uploading code expects to find them
python -m auditwheel repair -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" "${EXCLUDE_ARGS[@]}" dist/*
RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 "${package_type}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
