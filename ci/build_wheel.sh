#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2
package_type=$3

source rapids-configure-sccache
source rapids-date-string

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

rapids-generate-version > ./VERSION

cd "${package_dir}"

sccache --zero-stats

rapids-logger "Building '${package_name}' wheel"

python -m pip wheel \
    -w dist \
    -v \
    --no-deps \
    --disable-pip-version-check \
    --extra-index-url https://pypi.nvidia.com \
    .

sccache --show-adv-stats

case "${RAPIDS_CUDA_VERSION}" in
    12.*)
        EXCLUDE_ARGS=(
            --exclude "libcublas.so.12"
            --exclude "libcublasLt.so.12"
            --exclude "libcurand.so.10"
            --exclude "libcusolver.so.11"
            --exclude "libcusparse.so.12"
            --exclude "libnvJitLink.so.12"
        )
    ;;
    11.*)
        EXCLUDE_ARGS=()
    ;;
esac

case "${package_dir}" in
  python/pylibcugraph)
    EXCLUDE_ARGS+=(
      --exclude "libcugraph_c.so"
      --exclude "libcugraph.so"
    )
  ;;
  python/cugraph)
    EXCLUDE_ARGS+=(
      --exclude "libcugraph_c.so"
      --exclude "libcugraph.so"
    )
  ;;
esac

mkdir -p final_dist
python -m auditwheel repair -w final_dist "${EXCLUDE_ARGS[@]}" dist/*
RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 "${package_type}" final_dist
