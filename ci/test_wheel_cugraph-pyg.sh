#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -eoxu pipefail

package_name="cugraph-pyg"
package_dir="python/cugraph-pyg"

python_package_name=$(echo ${package_name}|sed 's/-/_/g')

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# use 'ls' to expand wildcard before adding `[extra]` requires for pip
RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist
# pip creates wheels using python package names
python -m pip install $(ls ./dist/${python_package_name}*.whl)[test]

# RAPIDS_DATASET_ROOT_DIR is used by test scripts
export RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"

if [[ "${CUDA_VERSION}" == "11.8.0" ]]; then
  rapids-logger "Installing PyTorch and PyG dependencies"
  PYTORCH_URL="https://download.pytorch.org/whl/cu118"
  rapids-retry python -m pip install torch==2.1.0 --index-url ${PYTORCH_URL}
  rapids-retry python -m pip install torch-geometric==2.4.0
  rapids-retry python -m pip install \
    pyg_lib \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

  rapids-logger "pytest cugraph-pyg (single GPU)"
  pushd python/cugraph-pyg/cugraph_pyg
  python -m pytest \
    --cache-clear \
    --ignore=tests/mg \
    tests
  popd
else
  rapids-logger "skipping cugraph-pyg wheel test on CUDA!=11.8"
fi
