#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -eoxu pipefail

package_name="cugraph-pyg"
package_dir="python/cugraph-pyg"

python_package_name=$(echo ${package_name}|sed 's/-/_/g')

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

librmm_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact rmm 1529 cpp)
rmm_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact rmm 1529 python)
libraft_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact raft 2264 cpp)
pylibraft_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact raft 2264 python)
libcugraphops_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact cugraph-ops 629 cpp e7c6f06)

RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./dist
RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist
RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-s3 python ./dist

# pip creates wheels using python package names
python -m pip install "${python_package_name}-${RAPIDS_PY_CUDA_SUFFIX}[test]"  --find-links ${librmm_wheelhouse} --find-links ${rmm_wheelhouse} --find-links ${libraft_wheelhouse} --find-links ${pylibraft_wheelhouse} --find-links ${libcugraphops_wheelhouse} --find-links ./dist

# RAPIDS_DATASET_ROOT_DIR is used by test scripts
export RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"

if [[ "${CUDA_VERSION}" == "11.8.0" ]]; then
  PYTORCH_URL="https://download.pytorch.org/whl/cu118"
  PYG_URL="https://data.pyg.org/whl/torch-2.1.0+cu118.html"
else
  PYTORCH_URL="https://download.pytorch.org/whl/cu121"
  PYG_URL="https://data.pyg.org/whl/torch-2.1.0+cu121.html"
fi
rapids-logger "Installing PyTorch and PyG dependencies"
rapids-retry python -m pip install torch==2.1.0 --index-url ${PYTORCH_URL}
rapids-retry python -m pip install torch-geometric==2.4.0
rapids-retry python -m pip install \
  ogb \
  pyg_lib \
  torch_scatter \
  torch_sparse \
  torch_cluster \
  torch_spline_conv \
  -f ${PYG_URL}

rapids-logger "pytest cugraph-pyg (single GPU)"
pushd python/cugraph-pyg/cugraph_pyg
python -m pytest \
  --cache-clear \
  --ignore=tests/mg \
  tests
# Test examples
for e in "$(pwd)"/examples/*.py; do
  rapids-logger "running example $e"
  (yes || true) | python $e
done
popd
