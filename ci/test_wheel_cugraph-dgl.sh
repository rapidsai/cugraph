#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -eoxu pipefail

package_name="cugraph-dgl"
package_dir="python/cugraph-dgl"

python_package_name=$(echo ${package_name}|sed 's/-/_/g')

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Download wheels built during this job.
RAPIDS_PY_WHEEL_NAME="pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-deps
RAPIDS_PY_WHEEL_NAME="cugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-deps
python -m pip install ./local-deps/*.whl

# use 'ls' to expand wildcard before adding `[extra]` requires for pip
RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-s3 ./dist
# pip creates wheels using python package names
python -m pip install $(ls ./dist/${python_package_name}*.whl)[test]


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
