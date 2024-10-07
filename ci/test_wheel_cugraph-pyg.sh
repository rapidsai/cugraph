#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -eoxu pipefail

package_name="cugraph-pyg"

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Download the pylibcugraph, cugraph, and cugraph-pyg built in the previous step
RAPIDS_PY_WHEEL_NAME="pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-deps
RAPIDS_PY_WHEEL_NAME="cugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-deps
RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-s3 ./dist

# determine pytorch and pyg sources
if [[ "${CUDA_VERSION}" == "11.8.0" ]]; then
  PYTORCH_URL="https://download.pytorch.org/whl/cu118"
  PYG_URL="https://data.pyg.org/whl/torch-2.3.0+cu118.html"
else
  PYTORCH_URL="https://download.pytorch.org/whl/cu121"
  PYG_URL="https://data.pyg.org/whl/torch-2.3.0+cu121.html"
fi

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install \
    -v \
    --extra-index-url "${PYTORCH_URL}" \
    --find-links "${PYG_URL}" \
    "$(echo ./local-deps/pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX}*.whl)" \
    "$(echo ./local-deps/cugraph_${RAPIDS_PY_CUDA_SUFFIX}*.whl)" \
    "$(echo ./dist/cugraph_pyg_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test]" \
    'dgl==2.4.0' \
    'ogb' \
    'pyg_lib' \
    'torch>=2.3.0,<2.4' \
    'torch-geometric>=2.5,<2.6' \
    'torch_scatter' \
    'torch_sparse'

# RAPIDS_DATASET_ROOT_DIR is used by test scripts
export RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"

# Used to skip certain examples in CI due to memory limitations
export CI_RUN=1

rapids-logger "pytest cugraph-pyg (single GPU)"
pushd python/cugraph-pyg/cugraph_pyg
python -m pytest \
  --cache-clear \
  --benchmark-disable \
  tests
# Test examples
for e in "$(pwd)"/examples/*.py; do
  rapids-logger "running example $e"
  (yes || true) | python $e
done
popd
