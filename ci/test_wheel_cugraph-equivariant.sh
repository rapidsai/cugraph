#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -eoxu pipefail

package_name="cugraph-equivariant"

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Download the cugraph-equivariant built in the previous step
RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-s3 ./dist

# determine pytorch source
PKG_CUDA_VER="$(echo ${CUDA_VERSION} | cut -d '.' -f1,2 | tr -d '.')"
PKG_CUDA_VER_MAJOR=${PKG_CUDA_VER:0:2}
if [[ "${PKG_CUDA_VER_MAJOR}" == "12" ]]; then
  PYTORCH_CUDA_VER="121"
else
  PYTORCH_CUDA_VER=$PKG_CUDA_VER
fi
PYTORCH_URL="https://download.pytorch.org/whl/cu${PYTORCH_CUDA_VER}"

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install \
    -v \
    --extra-index-url "${PYTORCH_URL}" \
    "$(echo ./dist/cugraph_equivariant_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test]" \
    'e3nn' \
    'torch>=2.3.0,<2.4'

python -m pytest python/cugraph-equivariant/cugraph_equivariant/tests
