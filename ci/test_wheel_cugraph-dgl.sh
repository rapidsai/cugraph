#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -eoxu pipefail

package_name="cugraph-dgl"
package_dir="python/cugraph-dgl"

python_package_name=$(echo ${package_name}|sed 's/-/_/g')

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

librmm_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact rmm 1529 cpp)
rmm_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact rmm 1529 python)
libraft_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact raft 2264 cpp 0435e6b)
pylibraft_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact raft 2264 python 0435e6b)
libcugraphops_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact cugraph-ops 629 cpp e7c6f06)

RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./dist
RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist
RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-s3 python ./dist

python -m pip install "${python_package_name}-${RAPIDS_PY_CUDA_SUFFIX}[test]>=0.0.0a0" --find-links ${librmm_wheelhouse} --find-links ${rmm_wheelhouse} --find-links ${libraft_wheelhouse} --find-links ${pylibraft_wheelhouse} --find-links ${libcugraphops_wheelhouse} --find-links ./dist

PKG_CUDA_VER="$(echo ${CUDA_VERSION} | cut -d '.' -f1,2 | tr -d '.')"
PKG_CUDA_VER_MAJOR=${PKG_CUDA_VER:0:2}
if [[ "${PKG_CUDA_VER_MAJOR}" == "12" ]]; then
  PYTORCH_CUDA_VER="121"
else
  PYTORCH_CUDA_VER=$PKG_CUDA_VER
fi
PYTORCH_URL="https://download.pytorch.org/whl/cu${PYTORCH_CUDA_VER}"
DGL_URL="https://data.dgl.ai/wheels/cu${PYTORCH_CUDA_VER}/repo.html"

# Starting from 2.2, PyTorch wheels depend on nvidia-nccl-cuxx>=2.19 wheel and
# dynamically link to NCCL. RAPIDS CUDA 11 CI images have an older NCCL version that
# might shadow the newer NCCL required by PyTorch during import (when importing
# `cupy` before `torch`).
if [[ "${NCCL_VERSION}" < "2.19" ]]; then
  PYTORCH_VER="2.1.0"
else
  PYTORCH_VER="2.3.0"
fi

rapids-logger "Installing PyTorch and DGL"
rapids-retry python -m pip install "torch==${PYTORCH_VER}" --index-url ${PYTORCH_URL}
rapids-retry python -m pip install dgl==2.0.0 --find-links ${DGL_URL}

python -m pytest python/cugraph-dgl/tests
